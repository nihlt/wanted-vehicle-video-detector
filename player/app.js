(() => {
  'use strict';

  // Дані згенеровано tools/process_clip.py: samples — рамки на кожному обробленому кадрі,
  // vehicles — машини (треки) з найкращим моментом для картки.
  const D = window.LIVE;
  // Перевірка номерів у базі розшуку: tools/wanted_check.py пише wanted.js поруч з data.js (файлу може й не бути).
  // result: 'wanted' — номер зараз у розшуку, 'was_wanted' — був раніше, 'clean' — немає, 'skipped' — не перевірено
  const WANTED = window.WANTED || {};
  const wantedOf = (v) => WANTED[v.n];
  const isMatch = (v) => ['wanted', 'was_wanted'].includes(wantedOf(v)?.result);
  const $ = (s) => document.querySelector(s);

  const video = $('#video');
  const strip = $('#strip');
  const lanesEl = $('#lanes');
  const timeline = $('#timeline');
  const inspector = $('#inspector');
  const overlay = $('#overlay');
  const live = $('#live');
  const tag = $('#tag');
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const W = D.source.width;
  const H = D.source.height;
  // Рамки — у пікселях кадру. У розмітці SVG стоїть 1920×1080, тож підганяємо їх і пропорції сцени під це відео
  for (const svg of [live, overlay]) svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  overlay.querySelectorAll('mask > rect:not(#maskHole), .dim').forEach((r) => {
    r.setAttribute('width', W);
    r.setAttribute('height', H);
  });
  $('#stage').style.setProperty('--ar', W / H);
  // Список — у порядку появи машин (як і номери №1, 2, 3…): тоді під час відтворення він їде лише вниз
  const all = [...D.vehicles].sort((a, b) => a.first_seen - b.first_seen || a.n - b.n);
  const byId = Object.fromEntries(all.map((v) => [v.id, v]));
  const byNum = Object.fromEntries(all.map((v) => [v.n, v]));
  const times = D.samples.map((s) => s[0]);
  let duration = D.source.duration;
  let filter = 'all';
  let selected = null;
  let visible = all;
  let cardEls = new Map();
  let passEls = new Map(); // смужка «скільки проїзду вже минуло» на кожній картці
  let segEls = new Map();
  let showLive = true;

  const TYPES = { car: 'Легкова', truck: 'Вантажівка', bus: 'Автобус', moto: 'Мото' };
  const fmt = (s) => {
    s = Math.max(0, s);
    return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  };
  const pct = (t) => (t / duration) * 100;
  const plural = (n) => {
    const a = n % 10, b = n % 100;
    if (a === 1 && b !== 11) return 'машина';
    if (a >= 2 && a <= 4 && (b < 12 || b > 14)) return 'машини';
    return 'машин';
  };

  const dmy = (s) => (s ? String(s).slice(0, 10).split('-').reverse().join('.') : '');
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // Номер машини — одне рішення з усіх кадрів її проїзду (tools/track_plates.py):
  // status 'read' — прочитано впевнено, 'uncertain' — прочитано, але кадри розмиті або не погоджуються
  function plateText(t) {
    let m = t.match(/^([A-Z]{2})(\d{4})([A-Z]{2})$/);
    if (m) return `${m[1]} ${m[2]} ${m[3]}`;
    m = t.match(/^(\d[A-Z]{2})(\d{4})$/);
    if (m) return `${m[1]} ${m[2]}`;
    m = t.match(/^([A-Z]{3})(\d{3})$/);
    if (m) return `${m[1]} ${m[2]}`;
    return t;
  }
  const isSure = (p) => p.status === 'read';
  const plateLabel = (p) => plateText(p.text) + (isSure(p) ? '' : '?');

  function plateHTML(v) {
    const p = v.plate;
    if (!p.text) {
      if (v.plate_crop) return `<img class="plate-img" src="${v.plate_crop}" alt="Номер машини №${v.n}, текст не прочитано">`;
      return '<span class="plate plate--none">Номер не знайдено</span>';
    }
    // синя смужка UA лише для звичайного українського формату AA 1234 BB (іменні й іноземні — без неї)
    const band = /^[A-Z]{2}\d{4}[A-Z]{2}$/.test(p.text) ? '<span class="plate-band" aria-hidden="true"><i></i>UA</span>' : '';
    return `<span class="plate${isSure(p) ? '' : ' plate--low'}">${band}<span class="plate-num">${esc(plateLabel(p))}</span></span>`;
  }

  /* ── рамки на відео ──────────────────────── */
  function nearestSample(t) {
    let lo = 0, hi = times.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] < t) lo = mid + 1; else hi = mid;
    }
    if (lo > 0 && t - times[lo - 1] < times[lo] - t) lo -= 1;
    return Math.abs(times[lo] - t) <= 0.05 ? D.samples[lo] : null;
  }

  // Рамки пораховано на кожному другому кадрі (30 разів на секунду), а відео йде 60 кадрів/с.
  // Між двома сусідніми порахованими кадрами ставимо рамку пропорційно часу — на тій самій лінії руху,
  // тому вона пливе разом з машиною на кожному кадрі відео, а не стоїть два кадри і стрибає.
  // Машина (чи її номер) лише на одному з двох кадрів — початок/кінець проїзду: показуємо ближчий, як є.
  function boxesAt(t) {
    let hi = 0, top = times.length;
    while (hi < top) {
      const mid = (hi + top) >> 1;
      if (times[mid] < t) hi = mid + 1; else top = mid;
    }
    const lo = hi - 1;
    if (hi < times.length && times[hi] - t < 1e-3) return D.samples[hi][1];
    if (lo < 0 || hi >= times.length || times[hi] - times[lo] > 0.05) {
      const s = nearestSample(t);
      return s ? s[1] : null;
    }
    const k = (t - times[lo]) / (times[hi] - times[lo]);
    const A = new Map(D.samples[lo][1].map((r) => [r[0], r]));
    const B = new Map(D.samples[hi][1].map((r) => [r[0], r]));
    const near = k < 0.5 ? A : B;
    const mix = (a, b, from, to) => a.slice(from, to).map((v, j) => v + (b[from + j] - v) * k);
    const out = [];
    for (const n of new Set([...A.keys(), ...B.keys()])) {
      const a = A.get(n), b = B.get(n), c = near.get(n);
      if (!a || !b) { if (c) out.push(c); continue; }
      const plate = a.length > 5 && b.length > 5 ? mix(a, b, 5, 9) : (c && c.length > 5 ? c.slice(5) : []);
      out.push([n, ...mix(a, b, 1, 5), ...plate]);
    }
    return out;
  }

  // Табличка з текстом номера над рамкою машини. Рішення прийняте з усього проїзду,
  // тому показуємо його з першого ж кадру, навіть коли номер ще далеко й нечитабельний
  const LP_FONT = 34;
  function plateLabelSVG(v, x, y, sel) {
    const p = v && v.plate;
    if (!p || !p.text) return '';
    const s = plateLabel(p);
    const tw = s.length * LP_FONT * 0.56, pw = tw + 20, ph = LP_FONT + 12;
    let ly = y - ph - 6, lx = x;
    if (ly < 4) { ly = y + 6; lx = x + 70; } // машина впритул до верхнього краю — табличка всередині рамки, правіше номера машини
    lx = Math.min(Math.max(4, lx), W - pw - 4);
    return `<g class="lp${isSure(p) ? '' : ' lp--low'}${sel}"><rect x="${lx}" y="${ly}" width="${pw}" height="${ph}" rx="6"/>`
      + `<text x="${lx + 10}" y="${ly + ph - 12}" textLength="${tw}" lengthAdjust="spacingAndGlyphs">${esc(s)}</text></g>`;
  }

  function renderLive(t) {
    const rows = showLive ? boxesAt(t) : null;
    if (!rows) { live.innerHTML = ''; return; }
    live.innerHTML = rows.map((b) => {
      const [n, x, y, w, h, px, py, pw, ph] = b;
      const sel = selected && selected.n === n ? ' sel' : '';
      const wd = WANTED[n]?.result === 'wanted' ? ' wanted' : '';
      let html = `<rect class="lb lb--car${sel}${wd}" x="${x * W}" y="${y * H}" width="${w * W}" height="${h * H}" rx="4"/>`;
      html += `<text class="lb-id${sel}" x="${x * W + 6}" y="${y * H + 30}">${n}</text>`;
      if (px !== undefined) html += `<rect class="lb lb--plate" x="${px * W}" y="${py * H}" width="${pw * W}" height="${ph * H}" rx="2"/>`;
      html += plateLabelSVG(byNum[n], x * W, y * H, sel + wd);
      return html;
    }).join('');
  }

  // Малюємо по кадру, який реально показано (mediaTime), а не по currentTime — так рамки не відстають від машин
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    const onFrame = (_, meta) => { renderLive(meta.mediaTime); video.requestVideoFrameCallback(onFrame); };
    video.requestVideoFrameCallback(onFrame);
  }

  $('#showLive').addEventListener('change', (e) => {
    showLive = e.target.checked;
    renderLive(video.currentTime);
  });

  /* ── стрічка ─────────────────────────────── */
  const WD_SHORT = { clean: 'не в розшуку', wanted: 'у розшуку', was_wanted: 'був у розшуку' };
  function wantedMeta(v) {
    const w = wantedOf(v);
    if (!w || !WD_SHORT[w.result]) return '';
    return `<span class="wd wd--${w.result}" title="База розшуку МВС через cardetect.in.ua, перевірено ${dmy(w.checked_at)}">${WD_SHORT[w.result]}</span>`;
  }

  function cardHTML(v) {
    const p = v.plate;
    const plateSay = p.text ? `номер ${p.text}${isSure(p) ? '' : ', прочитано непевно'}` : (v.plate_crop ? 'номер не прочитано' : 'номер не знайдено');
    const wd = WD_SHORT[wantedOf(v)?.result];
    const label = `Машина №${v.n}, з'являється на ${fmt(v.first_seen)}, ${TYPES[v.vehicle_type] || v.vehicle_type}, ${plateSay}${wd ? `, ${wd}` : ''}`;
    return `<li><button type="button" class="card" data-id="${v.id}" aria-label="${label}">
      <span class="thumb"><img src="${v.thumb}" alt="" loading="lazy"><b class="thumb-n">${v.n}</b></span>
      <span class="card-body">
        <span class="card-plate">${plateHTML(v)}</span>
        <span class="meta"><span class="num" title="Коли машина з'являється поруч">${fmt(v.first_seen)}</span><span>${TYPES[v.vehicle_type] || v.vehicle_type}</span>${wantedMeta(v)}</span>
      </span>
      <span class="pass" aria-hidden="true"><i></i></span>
    </button></li>`;
  }

  function renderStrip() {
    visible = all.filter((v) => filter === 'all' || (filter === 'wanted' ? isMatch(v) : v.plate.text));
    $('#count').textContent = visible.length === all.length
      ? `${all.length} ${plural(all.length)}`
      : `${visible.length} з ${all.length} ${plural(all.length)}`;
    inspector.remove(); // подробиці живуть усередині списку — виймаємо, щоб не стерти разом зі старими рядками
    strip.innerHTML = visible.map(cardHTML).join('');
    cardEls = new Map([...strip.querySelectorAll('.card')].map((el) => [el.dataset.id, el]));
    passEls = new Map([...cardEls].map(([id, el]) => [id, el.querySelector('.pass i')]));
    inFrame = new Set(); // картки й доріжки щойно створено без підсвітки — tick() підсвітить заново
    placeInspector();
    renderLanes();
    syncSelection(false);
    tick();
  }

  strip.addEventListener('click', (e) => {
    const card = e.target.closest('.card');
    if (!card) return;
    if (selected && selected.id === card.dataset.id) clearSelection(); // повторний клік згортає подробиці
    else select(byId[card.dataset.id]);
  });

  /* ── таймлайн ────────────────────────────── */
  function renderLanes() {
    // Машин багато і вони короткі, тому доріжок не більше MAX_LANES: якщо вільної немає,
    // відрізок лягає на ту, що звільнилась найраніше (може трохи перекритись)
    const ends = [];
    const PITCH = 6;
    const MAX_LANES = 6;
    const html = visible.map((v) => {
      let lane = ends.findIndex((end) => end <= v.first_seen - 0.2);
      if (lane < 0 && ends.length < MAX_LANES) { lane = ends.length; ends.push(0); }
      if (lane < 0) lane = ends.indexOf(Math.min(...ends));
      ends[lane] = v.last_seen;
      const best = ((v.best_time - v.first_seen) / (v.last_seen - v.first_seen)) * 100;
      return `<button type="button" class="lane-seg" data-id="${v.id}" tabindex="-1" title="Машина №${v.n}"
        style="left:${pct(v.first_seen)}%;width:${pct(v.last_seen - v.first_seen)}%;top:${lane * PITCH}px"><i style="left:${best}%"></i></button>`;
    }).join('');
    lanesEl.style.height = `${Math.max(1, ends.length) * PITCH}px`;
    lanesEl.innerHTML = html;
    segEls = new Map([...lanesEl.querySelectorAll('.lane-seg')].map((el) => [el.dataset.id, el]));
    for (const v of inFrame) segEls.get(v.id)?.classList.add('in'); // нові відрізки — без підсвітки
  }

  lanesEl.addEventListener('pointerdown', (e) => { if (e.target.closest('.lane-seg')) e.stopPropagation(); });
  lanesEl.addEventListener('click', (e) => {
    const seg = e.target.closest('.lane-seg');
    if (seg) select(byId[seg.dataset.id]);
  });

  let scrubbing = false;
  const seekFromEvent = (e) => {
    const r = timeline.getBoundingClientRect();
    video.currentTime = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * duration;
  };
  timeline.addEventListener('pointerdown', (e) => {
    scrubbing = true;
    timeline.setPointerCapture(e.pointerId);
    video.pause();
    seekFromEvent(e);
  });
  timeline.addEventListener('pointermove', (e) => { if (scrubbing) seekFromEvent(e); });
  timeline.addEventListener('pointerup', () => { scrubbing = false; });
  timeline.addEventListener('keydown', (e) => {
    const d = { PageUp: -5, PageDown: 5, Home: -Infinity, End: Infinity }[e.key];
    if (d === undefined) return;
    e.preventDefault();
    video.currentTime = Math.min(duration, Math.max(0, video.currentTime + d));
  });

  // Клік по рамці на відео на паузі вибирає цю машину
  $('#stageHit').addEventListener('click', (e) => {
    if (!video.paused) { video.pause(); return; }
    const r = e.currentTarget.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    const rows = boxesAt(video.currentTime);
    const hit = rows && rows.filter(([, x, y, w, h]) => fx >= x && fx <= x + w && fy >= y && fy <= y + h)
      .sort((a, b) => a[3] * a[4] - b[3] * b[4])[0];
    if (hit && byNum[hit[0]]) select(byNum[hit[0]]);
    else video.play();
  });

  /* ── вибір машини ────────────────────────── */
  function select(v, { scroll = true } = {}) {
    if (!v) return;
    selected = v;
    video.pause();
    if (Math.abs(video.currentTime - v.best_time) < 0.01) updateOverlay();
    else { hideOverlay(); video.currentTime = v.best_time; }
    // спершу переносимо подробиці під новий рядок, потім прокручуємо: інакше список зсувається вже після прокрутки
    renderInspector();
    syncSelection(scroll);
    renderLive(video.currentTime);
    setHash(`#${v.id}`);
  }

  function setHash(url) {
    try { history.replaceState(null, '', url); } catch { /* на file:// не критично */ }
  }

  function clearSelection() {
    selected = null;
    hideOverlay();
    syncSelection(false);
    renderInspector();
    renderLive(video.currentTime);
    setHash(location.pathname + location.search);
  }

  function syncSelection(scroll) {
    for (const [id, el] of cardEls) el.setAttribute('aria-current', String(selected?.id === id));
    for (const [id, el] of segEls) el.classList.toggle('sel', selected?.id === id);
    if (scroll && selected && cardEls.has(selected.id)) revealCard(cardEls.get(selected.id), true);
    updateNav();
  }

  // toTop: вибрану машину ставимо вгору списку, щоб під нею було видно подробиці;
  // інакше лише докручуємо, щоб рядок потрапив у видиму частину
  function revealCard(el, toTop) {
    const sr = strip.getBoundingClientRect();
    const er = el.getBoundingClientRect();
    let top;
    if (toTop) top = strip.scrollTop + er.top - sr.top - 8;
    else if (er.top < sr.top) top = strip.scrollTop + er.top - sr.top - 8;
    else if (er.bottom > sr.bottom) top = strip.scrollTop + er.bottom - sr.bottom + 8;
    else return;
    // вибір (клік, стрілки) — миттєво; плавно лише автопрокрутка під час відтворення
    strip.scrollTo({ top, behavior: toTop || reduceMotion ? 'auto' : 'smooth' });
  }

  function neighbours() {
    const i = selected ? visible.indexOf(selected) : -1;
    if (i >= 0) return { prev: visible[i - 1], next: visible[i + 1] };
    const t = video.currentTime;
    return {
      prev: [...visible].reverse().find((v) => v.first_seen < t - 0.01),
      next: visible.find((v) => v.first_seen > t + 0.01),
    };
  }
  function updateNav() {
    const n = neighbours();
    $('#prev').disabled = !n.prev;
    $('#next').disabled = !n.next;
  }
  $('#prev').addEventListener('click', () => select(neighbours().prev));
  $('#next').addEventListener('click', () => select(neighbours().next));

  /* ── підсвітка обраної машини на паузі ───── */
  const setRect = (el, b) => {
    el.setAttribute('x', b[0] * W); el.setAttribute('y', b[1] * H);
    el.setAttribute('width', b[2] * W); el.setAttribute('height', b[3] * H);
  };
  function showOverlay() {
    const v = selected;
    setRect($('#maskHole'), v.bbox);
    setRect($('#carBox'), v.bbox);
    const pb = $('#plateBox');
    if (v.plate.bbox) { setRect(pb, v.plate.bbox); pb.style.display = ''; } else pb.style.display = 'none';
    tag.innerHTML = `<span class="tag-n">№${v.n}</span>${plateHTML(v)}`;
    tag.style.left = `${v.bbox[0] * 100}%`;
    tag.style.top = `${v.bbox[1] * 100}%`;
    // машина біля правого краю: табличку з номером зсуваємо всередину кадру, щоб не обрізалась
    const stageW = tag.parentElement.clientWidth;
    if (tag.offsetLeft + tag.offsetWidth > stageW - 4) tag.style.left = `${Math.max(4, stageW - tag.offsetWidth - 4)}px`;
    tag.classList.toggle('inside', v.bbox[1] < 0.1);
    overlay.classList.add('on');
    tag.classList.add('on');
    live.classList.add('off');
  }
  function hideOverlay() {
    overlay.classList.remove('on');
    tag.classList.remove('on');
    live.classList.remove('off');
  }
  function updateOverlay() {
    if (selected && video.paused && !video.seeking && Math.abs(video.currentTime - selected.best_time) < 0.06) showOverlay();
    else hideOverlay();
  }

  /* ── інспектор ───────────────────────────── */
  // Подробиці стоять одразу під рядком вибраної машини; якщо її сховано фільтром — над списком
  function placeInspector() {
    const el = selected && cardEls.get(selected.id);
    if (el) el.parentElement.appendChild(inspector);
    else strip.before(inspector);
  }

  // Збіг у базі — ще не доказ: OCR міг помилитись на символ і влучити в чужий номер, тому показуємо марку й колір з бази
  function wantedHTML(v) {
    const w = wantedOf(v);
    if (!w) return '';
    if (w.result === 'skipped') return '<p class="ins-conf" title="Перевіряються лише номери формату АА 1234 АА"><span>База розшуку</span><b>не перевірено</b></p>';
    if (w.result === 'clean') return `<p class="ins-conf" title="База розшуку МВС через cardetect.in.ua, перевірено ${dmy(w.checked_at)}"><span>База розшуку</span><b>немає</b></p>`;
    const car = [w.brand, w.color && w.color.toLowerCase()].filter(Boolean).join(', ');
    return `<div class="ins-wanted ins-wanted--${w.result}">
        <b>${w.result === 'wanted' ? 'Номер у базі розшуку' : 'Номер був у розшуку раніше'}</b>
        ${car ? `<span>${esc(car)}</span>` : ''}
        ${w.theft_date ? `<span>викрадено ${dmy(w.theft_date)}</span>` : ''}
        ${w.organ ? `<span class="ins-wanted-organ">${esc(w.organ)}</span>` : ''}
        <small>Звірте марку з фото: номер міг бути прочитаний з помилкою</small>
      </div>`;
  }

  function renderInspector() {
    $('#hint').hidden = Boolean(selected);
    inspector.hidden = !selected;
    if (!selected) { inspector.innerHTML = ''; return; }
    placeInspector();
    const v = selected;
    const p = v.plate;
    // Лише те, чого немає на картці: скільки читань збіглося, найчіткіша вирізка номера і кнопка.
    // Тип, час і «непевно» (знак «?») уже видно в рядку списку
    inspector.innerHTML = `
      ${p.text ? `<p class="ins-conf"><span>Читань, що збіглися</span><b class="${isSure(p) ? '' : 'conf-low'}">${p.n_agree} з ${p.n_reads}</b></p>` : ''}
      ${wantedHTML(v)}
      ${v.plate_crop ? `<figure class="ins-crop"><img src="${v.plate_crop}" alt="Номер машини №${v.n} на кадрі"></figure>` : ''}
      <button type="button" class="btn" id="playFrom">Відтворити проїзд</button>`;
    $('#playFrom').addEventListener('click', () => {
      video.currentTime = v.first_seen;
      video.play();
    });
  }

  /* ── відтворення ─────────────────────────── */
  let lastAuto = null;
  let userScrolledAt = 0;
  let inFrame = new Set(); // машини, чиї картки зараз підсвічені як «у кадрі»
  const maxPass = Math.max(0, ...all.map((v) => v.last_seen - v.first_seen));
  // скільки машин зі списку з'явились не пізніше t (бінарний пошук: visible упорядкований за появою)
  function upperBound(t) {
    let lo = 0, hi = visible.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (visible[mid].first_seen <= t) lo = mid + 1; else hi = mid;
    }
    return lo;
  }
  strip.addEventListener('wheel', () => { userScrolledAt = performance.now(); }, { passive: true });
  strip.addEventListener('pointerdown', () => { userScrolledAt = performance.now(); });

  function tick() {
    const t = video.currentTime;
    $('#fill').style.width = `${pct(t)}%`;
    $('#head').style.left = `${pct(t)}%`;
    $('#tc').textContent = fmt(t);
    timeline.setAttribute('aria-valuenow', Math.round(t));
    timeline.setAttribute('aria-valuetext', fmt(t));
    // Машини в кадрі: список упорядкований за появою, тож дивимось лише ті, що з'явились не раніше,
    // ніж за найдовший проїзд до t (на всьому відео — кілька десятків замість сотень щокадру)
    const now = new Set();
    let newest = null;
    for (let i = upperBound(t) - 1; i >= 0 && visible[i].first_seen >= t - maxPass; i--) {
      const v = visible[i];
      if (t > v.last_seen) continue;
      now.add(v);
      if (!newest) newest = v;
      passEls.get(v.id)?.style.setProperty('width', `${((t - v.first_seen) / (v.last_seen - v.first_seen)) * 100}%`);
    }
    for (const v of inFrame) {
      if (now.has(v)) continue;
      cardEls.get(v.id)?.classList.remove('in');
      passEls.get(v.id)?.style.setProperty('width', '0');
      segEls.get(v.id)?.classList.remove('in');
    }
    for (const v of now) {
      if (inFrame.has(v)) continue;
      cardEls.get(v.id)?.classList.add('in');
      segEls.get(v.id)?.classList.add('in');
    }
    inFrame = now;
    // Список синхронізуємо з машиною, яку виявили останньою, і лише раз — коли вона з'явилась.
    // Далі він стоїть, поки не з'явиться ще новіша: назад не стрибає, і машина, що їде поруч усю
    // хвилину (Kangoo №17), його не тримає
    if (!video.paused && newest && (!lastAuto || newest.first_seen > lastAuto.first_seen)
        && performance.now() - userScrolledAt > 4000) {
      lastAuto = newest;
      const el = cardEls.get(newest.id);
      if (el) revealCard(el, false);
    }
  }

  function loop() {
    tick();
    if (!video.paused) requestAnimationFrame(loop);
  }

  video.addEventListener('loadedmetadata', () => {
    duration = video.duration || duration;
    $('#dur').textContent = fmt(duration);
    timeline.setAttribute('aria-valuemax', Math.round(duration));
    renderLanes();
    tick();
  });
  video.addEventListener('play', () => {
    document.body.classList.add('playing');
    $('#play').setAttribute('aria-label', 'Пауза');
    hideOverlay();
    lastAuto = null;
    requestAnimationFrame(loop);
  });
  video.addEventListener('pause', () => {
    document.body.classList.remove('playing');
    $('#play').setAttribute('aria-label', 'Відтворити');
    updateOverlay();
    updateNav();
  });
  video.addEventListener('seeking', () => { lastAuto = null; }); // перемотали — синхронізуємось заново з нового місця
  video.addEventListener('seeked', () => { tick(); updateOverlay(); updateNav(); renderLive(video.currentTime); });
  video.addEventListener('timeupdate', () => {
    tick();
    if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype)) renderLive(video.currentTime);
  });

  const togglePlay = () => (video.paused ? video.play() : video.pause());
  $('#play').addEventListener('click', togglePlay);

  /* ── фільтр ──────────────────────────────── */
  document.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => {
    filter = b.dataset.filter;
    document.querySelectorAll('.seg button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    renderStrip();
  }));

  /* ── клавіатура ──────────────────────────── */
  document.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement && e.target.type !== 'checkbox') return;
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === 'ArrowRight') { e.preventDefault(); select(neighbours().next); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); select(neighbours().prev); }
    else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    else if (e.key === 'Escape') clearSelection();
  });

  /* ── старт ───────────────────────────────── */
  const nMatch = all.filter(isMatch).length;
  const wantedBtn = document.querySelector('.seg [data-filter="wanted"]');
  if (wantedBtn && nMatch) { wantedBtn.hidden = false; wantedBtn.textContent = `У розшуку: ${nMatch}`; }
  $('#dur').textContent = fmt(duration);
  renderStrip();
  renderInspector();
  const fromHash = byId[location.hash.slice(1)];
  if (fromHash) select(fromHash);
  else renderLive(0);
})();
