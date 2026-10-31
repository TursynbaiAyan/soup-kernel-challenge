// Builds docs/Soup-Kernel-Brightly.pptx from docs/results.json: `node docs/build_deck.cjs`.
// When results.json has a "kaggle" block, the results slide shows the T4 numbers.
const path = require("path");
const pptxgen = require("pptxgenjs");
// The theme writer from the pptx skill, when available (sets the deck's scheme colors).
const applyTheme = process.env.PPTX_SKILL
  ? require(path.join(process.env.PPTX_SKILL, "scripts", "apply_theme.js")).applyTheme
  : async () => {};

const R = require("./results.json");
const OUT = path.join(__dirname, "Soup-Kernel-Brightly.pptx");

const THEME = {
  name: "Soup Kernel",
  headFontFace: "Arial",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0E1116", lt1: "FFFFFF", dk2: "1B2230", lt2: "EEF1F5",
    accent1: "F5A524", accent2: "2EC4B6", accent3: "7C8799", accent4: "E4572E",
    accent5: "9BE564", accent6: "5B6CFF", hlink: "2EC4B6", folHlink: "7C8799",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Soup Kernel Challenge — Brightly Production";
pres.author = R.team;
const C = pres.SchemeColor;

// ---- layouts -------------------------------------------------------------------------------
const footer = (dark) => ({
  text: { text: `${R.team} · Decentrathon 6.0 · Soup Kernel Challenge`,
    options: { x: 0.5, y: 5.2, w: 7, h: 0.25, fontSize: 9, color: dark ? C.accent3 : C.accent3 } },
});
pres.defineSlideMaster({
  title: "DARK_TITLE", background: { color: HEX.dk1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.55, w: 8.8, h: 1.5,
    fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
  { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.15, w: 8.8, h: 0.9,
    fontSize: 18, color: C.accent2, valign: "top", align: "left" }, text: "" } }],
});
pres.defineSlideMaster({
  title: "LIGHT_CONTENT", background: { color: HEX.lt1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.75,
    fontSize: 28, bold: true, color: C.text1, valign: "middle", align: "left" }, text: "" } }, footer(false)],
  slideNumber: { x: 9.1, y: 5.2, w: 0.4, h: 0.25, fontSize: 9, color: C.accent3 },
});
pres.defineSlideMaster({
  title: "DARK_CONTENT", background: { color: HEX.dk1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.75,
    fontSize: 28, bold: true, color: C.background1, valign: "middle", align: "left" }, text: "" } }, footer(true)],
  slideNumber: { x: 9.1, y: 5.2, w: 0.4, h: 0.25, fontSize: 9, color: C.accent3 },
});

// ---- helpers -------------------------------------------------------------------------------
let n = 0;
const id = (s) => `${s}-${++n}`;
function text(slide, t, o) {
  slide.addText(t, { isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top",
    objectName: id("text"), ...o });
}
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
    fill: { color: fill }, line: { type: "none" }, objectName: id("card") });
}
function chip(slide, x, y, label, fill, color) { // numbered circle: the deck's motif
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.42, h: 0.42, fill: { color: fill },
    line: { type: "none" }, objectName: id("chip") });
  text(slide, label, { x, y, w: 0.42, h: 0.42, fontSize: 14, bold: true, color, align: "center",
    valign: "middle" });
}
function stat(slide, x, y, w, big, label, bigColor, labelColor) {
  text(slide, big, { x, y, w, h: 0.9, fontSize: 44, bold: true, color: bigColor,
    fontFace: THEME.headFontFace });
  text(slide, label, { x, y: y + 0.92, w, h: 0.6, fontSize: 13, color: labelColor });
}
const ru = (v) => typeof v === "number" ? v.toLocaleString("ru-RU", { maximumFractionDigits: 4 }) : v;
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt" };

// ---- 1. title ------------------------------------------------------------------------------
pres.addSection({ title: "Начало" });
let s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Начало" });
s.addText("Голова модели без логитов", { placeholder: "title" });
s.addText("Слитый lm_head + cross-entropy на Triton для soup train на Kaggle T4",
  { placeholder: "body" });
chip(s, 0.6, 0.6, "3", HEX.accent1, C.text1);
text(s, "своих @triton.jit-кернела", { x: 1.15, y: 0.62, w: 4, h: 0.4, fontSize: 14,
  color: C.background1, valign: "middle" });
text(s, `${R.team} · Decentrathon 6.0`, { x: 0.6, y: 4.75, w: 6, h: 0.35, fontSize: 13,
  color: C.accent3 });
s.addNotes("Мы ускоряли и облегчали голову модели: lm_head и функцию потерь. " +
  "Три собственных Triton-кернела, подключённых к Soup одной функцией apply().");

// ---- 2. problem ----------------------------------------------------------------------------
pres.addSection({ title: "Задача" });
s = pres.addSlide({ masterName: "LIGHT_CONTENT", sectionTitle: "Задача" });
s.addText("T4 упирается в память, а не в модель", { placeholder: "title" });
stat(s, 0.5, 1.35, 2.8, "14,3 ГБ", "пик памяти базового Soup из 15 ГБ на T4", C.accent4, C.text2);
stat(s, 3.55, 1.35, 2.8, "≈ 80 с", "весь soup train: 64 шага, Qwen2.5-1.5B NF4, LoRA r=16", C.text1, C.text2);
stat(s, 6.6, 1.35, 2.9, "152 тыс.", "токенов в словаре: логиты [4 × 512 × 151 936]", C.accent1, C.text2);
card(s, 0.5, 3.35, 9, 1.55, HEX.lt2);
text(s, [
  { text: "Задача кейса. ", options: { bold: true } },
  { text: "Тот же запуск обучения — быстрее или с меньшей памятью, адаптер того же качества. " +
    "Свои Triton-кернелы ≥ 5% времени обучения; loss, энтропия, точность и rewind-лог — в пределах 1–3% от базы." },
], { x: 0.75, y: 3.55, w: 8.5, h: 1.2, fontSize: 15, color: C.text1 });
s.addNotes("Исходные цифры из ТЗ: около 80 секунд и 14 из 15 гигабайт. Словарь Qwen огромный, " +
  "поэтому логиты — главный потребитель памяти.");

// ---- 3. where time and memory go -----------------------------------------------------------
s = pres.addSlide({ masterName: "LIGHT_CONTENT", sectionTitle: "Задача" });
s.addText("Куда уходит шаг обучения", { placeholder: "title" });
s.addChart(pres.charts.DOUGHNUT, [{ name: "Шаг", labels: ["Поэлементные операции и касты",
  "Умножение матриц", "Softmax / cross-entropy", "Attention", "Распаковка NF4"],
  values: [44, 36, 7, 5, 5] }], {
  x: 0.4, y: 1.1, w: 4.6, h: 3.9, holeSize: 55, showPercent: false, showValue: true, dataLabelFormatCode: '0"%"', showLegend: true,
  legendPos: "b", legendFontSize: 10, legendFontFace: "+mn-lt", dataLabelColor: "FFFFFF",
  dataLabelFontSize: 11, chartColors: [HEX.accent3, "A9B1BE", HEX.accent1, "C9CFD8", "DDE1E7"],
  showTitle: true, title: "GPU-время шага, % (профиль базы из ТЗ)", titleFontSize: 12,
  titleColor: HEX.dk2, ...chartText, objectName: id("chart"),
});
const rows = [["Логиты fp16 [4, T, V]", "0,6 ГБ"], ["Их fp32-копия для loss", "1,2 ГБ"],
  ["Буферы softmax и градиента", "≈1,2 ГБ"], ["fp16-копия веса lm_head", "0,47 ГБ"]];
text(s, "Хвост модели при T = 512", { x: 5.4, y: 1.2, w: 4.1, h: 0.4, fontSize: 16, bold: true });
rows.forEach(([k, v], i) => {
  card(s, 5.4, 1.7 + i * 0.62, 4.1, 0.52, HEX.lt2);
  text(s, k, { x: 5.6, y: 1.7 + i * 0.62, w: 2.8, h: 0.52, fontSize: 13, valign: "middle" });
  text(s, v, { x: 8.3, y: 1.7 + i * 0.62, w: 1.05, h: 0.52, fontSize: 14, bold: true,
    color: C.accent4, align: "right", valign: "middle" });
});
text(s, "Голова заморожена: логиты нужны только loss и метрикам.", { x: 5.4, y: 4.25, w: 4.1,
  h: 0.6, fontSize: 13, italic: true, color: C.text2 });

// ---- 4. idea: base vs ours -----------------------------------------------------------------
pres.addSection({ title: "Решение" });
s = pres.addSlide({ masterName: "LIGHT_CONTENT", sectionTitle: "Решение" });
s.addText("Идея: не создавать логиты вообще", { placeholder: "title" });
const cols = [
  ["База Soup", HEX.lt2, C.text1, ["Каст fp32-веса lm_head → fp16 (467 МБ)",
    "Логиты для всех позиций, включая pad", "Апкаст в fp32 и буферы softmax/градиента",
    "TRL и rewind заново читают полные логиты"]],
  ["Наше решение", HEX.dk1, C.background1, ["Вес читается в fp32, округляется в регистрах",
    "Только реальные позиции, чанки по 1 024", "Loss, энтропия, argmax и градиент за один проход",
    "Метрики TRL и rewind — из значений кернелов"]],
];
cols.forEach(([head, fill, color, items], ci) => {
  const x = 0.5 + ci * 4.6;
  card(s, x, 1.2, 4.4, 3.75, fill);
  text(s, head, { x: x + 0.3, y: 1.4, w: 3.8, h: 0.45, fontSize: 18, bold: true,
    color: ci ? C.accent1 : C.text1 });
  items.forEach((t, i) => {
    chip(s, x + 0.3, 2.0 + i * 0.7, String(i + 1), ci ? HEX.accent1 : HEX.accent3,
      ci ? C.text1 : C.background1);
    text(s, t, { x: x + 0.85, y: 2.0 + i * 0.7, w: 3.35, h: 0.55, fontSize: 13, color,
      valign: "middle" });
  });
});

// ---- 5. three kernels ----------------------------------------------------------------------
s = pres.addSlide({ masterName: "DARK_CONTENT", sectionTitle: "Решение" });
s.addText("Три кернела на чанк реальных позиций", { placeholder: "title" });
const kernels = [
  ["lm_head_logits_kernel", "Логиты = h · Wᵀ",
    "W читается в fp32 и округляется до fp16 в регистрах, как у autocast. Тайлы идут по строкам, чтобы кусок W брался из L2."],
  ["ce_stats_grad_kernel", "Online-softmax по словарю",
    "Max, Σe, Σe·x → log-sum-exp, энтропия, argmax, CE метки. Второй проход пишет (softmax − onehot)·256 на место логитов."],
  ["lm_head_dgrad_kernel", "dh = G · W, split-K",
    "Свёртка по 151 936 токенам делится между блоками, частичные суммы — atomic_add в fp32. Только позиции с меткой."],
];
kernels.forEach(([name, what, how], i) => {
  const x = 0.5 + i * 3.05;
  card(s, x, 1.25, 2.85, 3.0, HEX.dk2);
  chip(s, x + 0.25, 1.45, String(i + 1), HEX.accent1, C.text1);
  text(s, what, { x: x + 0.25, y: 2.0, w: 2.4, h: 0.65, fontSize: 16, bold: true,
    color: C.background1 });
  text(s, name, { x: x + 0.25, y: 2.65, w: 2.4, h: 0.3, fontSize: 10, color: C.accent2,
    fontFace: "Courier New" });
  text(s, how, { x: x + 0.25, y: 3.05, w: 2.4, h: 1.75, fontSize: 12, color: "C9CFD8" });
});

// ---- 6. engineering details ----------------------------------------------------------------
s = pres.addSlide({ masterName: "LIGHT_CONTENT", sectionTitle: "Решение" });
s.addText("Детали, от которых зависит результат", { placeholder: "title" });
const details = [
  ["Градиент в forward", "Голова заморожена: dh считается сразу, backward только умножает на масштаб GradScaler."],
  ["Множитель 256", "Малые вероятности не уходят в ноль fp16; множитель снимается в fp32."],
  ["Pad отброшен", "Считаются только позиции с attention_mask = 1; строки батчей не смешиваются."],
  ["Одна компиляция", "Фиксированные тайлы, do_not_specialize: каждый кернел компилируется один раз."],
  ["Память T4", "24–32 КБ разделяемой памяти на блок при пределе 64 КБ."],
  ["Без подмены режимов", "Встроенные режимы Soup выключены; патч — Qwen2 forward и compute_loss TRL."],
];
details.forEach(([h, d], i) => {
  const x = 0.5 + (i % 2) * 4.6, y = 1.2 + Math.floor(i / 2) * 1.27;
  card(s, x, y, 4.4, 1.12, HEX.lt2);
  text(s, h, { x: x + 0.25, y: y + 0.13, w: 3.9, h: 0.35, fontSize: 15, bold: true });
  text(s, d, { x: x + 0.25, y: y + 0.5, w: 3.9, h: 0.55, fontSize: 12, color: C.text2 });
});

// ---- 7. results ----------------------------------------------------------------------------
pres.addSection({ title: "Результаты" });
const L = R.local, K = R.kaggle;
s = pres.addSlide({ masterName: "LIGHT_CONTENT", sectionTitle: "Результаты" });
if (K) {
  s.addText("Результат на Kaggle T4", { placeholder: "title" });
  stat(s, 0.5, 1.3, 2.9, `${ru(+K.speedup.toFixed(2))}×`, `быстрее: ${K.time_base} с → ${K.time_ours} с, весь soup train`, C.accent2, C.text2);
  stat(s, 3.55, 1.3, 2.9, `${ru(+K.memory.toFixed(2))}×`, `меньше памяти: ${K.mem_base} → ${K.mem_ours} МиБ (nvidia-smi)`, C.accent1, C.text2);
  stat(s, 6.6, 1.3, 2.9, `${K.eval_pct >= 0 ? "+" : ""}${ru(+K.eval_pct.toFixed(2))}%`, "eval loss против базы (порог +0,5%)", C.text1, C.text2);
  card(s, 0.5, 3.45, 9, 1.45, HEX.lt2);
  text(s, [{ text: "Проверки check_rules.py: ", options: { bold: true } },
    { text: `${K.checks}. Доля наших кернелов в обучении: ${K.share}% (порог 5%).` }],
  { x: 0.75, y: 3.65, w: 8.5, h: 1.1, fontSize: 15 });
} else {
  s.addText("Голова: в 2,3 раза быстрее, в 13 раз легче", { placeholder: "title" });
  const bar = (title, labels, base, ours, unit, x) => s.addChart(pres.charts.BAR, [
    { name: title, labels, values: [base, ours] }], {
    x, y: 1.15, w: 4.4, h: 3.2, barDir: "col", chartColors: [HEX.accent3, HEX.accent1],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelColor: HEX.dk2,
    dataLabelFormatCode: `#,##0" ${unit}"`, showLegend: false, showTitle: true, title,
    titleFontSize: 13, titleColor: HEX.dk2, catAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 12,
    valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: Math.ceil(base * 1.3), valGridLine: { style: "none" }, catGridLine: { style: "none" },
    ...chartText, objectName: id("chart") });
  bar("Время forward + backward головы", ["Эталон (autocast)", "Наши кернелы"],
    L.head_ms_base, L.head_ms_ours, "мс", 0.5);
  bar("Дополнительный пик памяти", ["Эталон (autocast)", "Наши кернелы"],
    L.head_mib_base, L.head_mib_ours, "МиБ", 5.1);
  text(s, `Замер: ${L.gpu}, 4 строки до 400 токенов (1 160 позиций), голова 1 536 × 151 936. ` +
    `Логиты ${ru(L.logits_ms)} мс против ${ru(L.logits_cublas_ms)} у cuBLAS fp16 — хотя мы читаем fp32-веса.`,
  { x: 0.5, y: 4.45, w: 9, h: 0.6, fontSize: 12, color: C.text2 });
}

// ---- 8. correctness ------------------------------------------------------------------------
s = pres.addSlide({ masterName: "LIGHT_CONTENT", sectionTitle: "Результаты" });
s.addText("То же обучение: метрики совпадают с базой", { placeholder: "title" });
const head = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 } } });
s.addTable([
  [head("Величина"), head("Отличие от базы"), head("Допуск")],
  ["Loss, первый шаг", `${ru(L.loss_first_pct)}%`, "1%"],
  ["Loss, среднее по шагам", `${ru(L.loss_mean_pct)}%`, "3%"],
  ["Энтропия, точность, число токенов", "0,0000%", "3%"],
  ["Loss строк в rewind.jsonl", `≤ ${ru(L.rewind_max_pct)}%`, "3%"],
  ["Пик памяти PyTorch (сквозной прогон)", `${ru(L.e2e_peak_base)} → ${ru(L.e2e_peak_ours)} МиБ`, "—"],
], { x: 0.5, y: 1.2, w: 5.6, colW: [2.9, 1.65, 1.05], fontSize: 12, fontFace: "+mn-lt",
  color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "D5DAE1" }, rowH: 0.42,
  objectName: id("table") });
card(s, 6.4, 1.2, 3.1, 1.6, HEX.dk1);
text(s, String(L.soup_tests), { x: 6.65, y: 1.3, w: 2.6, h: 0.85, fontSize: 48, bold: true,
  color: C.accent2 });
text(s, "тестов Soup проходят с нашими патчами", { x: 6.65, y: 2.15, w: 2.6, h: 0.55,
  fontSize: 12, color: C.background1 });
card(s, 6.4, 3.0, 3.1, 1.45, HEX.lt2);
text(s, "Сквозной прогон: TRL SFTTrainer + RewindSFTTrainer Soup, LoRA r=16 на q/v, fp16, 12 шагов. " +
  "Кернелы проверены против PyTorch, включая реальный размер головы.",
{ x: 6.6, y: 3.15, w: 2.75, h: 1.5, fontSize: 11, color: C.text2 });

// ---- 9. limits and next steps --------------------------------------------------------------
pres.addSection({ title: "Дальше" });
s = pres.addSlide({ masterName: "DARK_CONTENT", sectionTitle: "Дальше" });
s.addText("Ограничения и следующие шаги", { placeholder: "title" });
const lim = ["Ускорена только голова: 44% поэлементных операций остаются как в базе",
  "Pad отбрасывается только в голове, декодер считает его по-прежнему",
  "atomic_add делает градиент недетерминированным на уровне округления",
  "Тайлы подобраны на Ampere с запасом по памяти Turing"];
const next = ["Слитый RMSNorm + приведение к fp16", "Слитая LoRA-добавка в q_proj и v_proj",
  "Распаковка NF4 прямо внутри умножения", "Удаление pad во всём декодере"];
[["Ограничения", lim, HEX.accent3], ["Что дальше", next, HEX.accent1]].forEach(([h, items, col], ci) => {
  const x = 0.5 + ci * 4.6;
  text(s, h, { x, y: 1.2, w: 4.3, h: 0.45, fontSize: 18, bold: true,
    color: ci ? C.accent1 : C.background1 });
  items.forEach((t, i) => {
    chip(s, x, 1.8 + i * 0.78, String(i + 1), col, ci ? C.text1 : C.background1);
    text(s, t, { x: x + 0.55, y: 1.8 + i * 0.78, w: 3.75, h: 0.62, fontSize: 13,
      color: "D5DAE1", valign: "middle" });
  });
});

// ---- 10. close -----------------------------------------------------------------------------
s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Дальше" });
s.addText("Тот же адаптер. Меньше памяти. Быстрее.", { placeholder: "title" });
s.addText(R.repo, { placeholder: "body" });
text(s, "Идеи: Liger-Kernel FLCE, Cut Cross-Entropy (arXiv 2411.09009), online softmax " +
  "(arXiv 1805.02867), туториал Triton по matmul. Код написан нами.",
{ x: 0.6, y: 4.5, w: 8.8, h: 0.6, fontSize: 11, color: C.accent3 });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
