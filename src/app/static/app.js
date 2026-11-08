"use strict";

const distances = ["24,1", "25,3", "26,5", "27,7"];
const inputs = [];
const form = document.querySelector("#prediction");
const run = document.querySelector("#run");
const example = document.querySelector("#example");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
const result = document.querySelector("#result");
let busy = false;

function sync(message) {
  const count = inputs.filter((item) => item.data).length;
  run.disabled = busy || count !== distances.length;
  example.disabled = busy;
  for (const item of inputs) item.field.disabled = busy;
  status.textContent = message ?? ("Загружено " + count + " из 4");
}

function showError(message = "") {
  error.textContent = message;
  error.hidden = !message;
}

function setImage(item, url, filename) {
  item.data = url ? url.split(",", 2)[1] : null;
  if (url) item.image.src = url;
  else item.image.removeAttribute("src");
  item.image.hidden = !url;
  item.prompt.hidden = Boolean(url);
  item.filename.textContent = filename;
  result.hidden = true;
}

async function request(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Ошибка локального сервера.");
  return body;
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Не удалось прочитать PNG."));
    reader.readAsDataURL(file);
  });
}

async function withStatus(message, action) {
  busy = true;
  showError();
  sync(message);
  let completed;
  try {
    completed = await action();
  } catch (err) {
    showError(err.message);
  } finally {
    busy = false;
    sync(completed);
  }
}

for (const distance of distances) {
  const card = document.querySelector("#input-card").content.cloneNode(true);
  const item = {
    field: card.querySelector("input"),
    image: card.querySelector("img"),
    prompt: card.querySelector(".frame span"),
    filename: card.querySelector(".filename"),
    data: null,
  };
  card.querySelector("strong").textContent = distance + " мм";
  item.image.alt = "Цель на " + distance + " мм";
  item.field.setAttribute("aria-label", item.image.alt);
  inputs.push(item);
  item.field.addEventListener("change", () => {
    const file = item.field.files[0];
    if (!file) return;
    setImage(item, null, "");
    withStatus("Читаю PNG…", async () => {
      if (file.size > 128 * 1024) {
        item.field.value = "";
        throw new Error("Один PNG должен быть не больше 128 КБ.");
      }
      setImage(item, await readFile(file), file.name);
    });
  });
  document.querySelector("#inputs").append(card);
}

example.addEventListener("click", () => withStatus("Загружаю пример…", async () => {
  const data = await request("/api/example");
  data.images.forEach((url, index) => {
    inputs[index].field.value = "";
    setImage(inputs[index], url, "Пример · 0" + (index + 1) + ".png");
  });
}));

function format(value) {
  return value.toLocaleString("ru", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
}

function render(data) {
  document.querySelector("#doe").src = data.doe;
  document.querySelector("#download").href = data.doe;
  document.querySelector("#summary").textContent =
    "Среднее по четырём плоскостям · SSIM " + format(data.mean.ssim) +
    " · Pearson " + format(data.mean.pearson) +
    " · Синтез " + format(data.synthesis_ms) + " мс";
  const grid = document.querySelector("#planes");
  grid.replaceChildren();
  data.planes.forEach((plane, index) => {
    const card = document.querySelector("#output-card").content.cloneNode(true);
    const distance = distances[index] + " мм";
    card.querySelector("h4").textContent = distance;
    const image = card.querySelector("img");
    image.src = plane.image;
    image.alt = "Реконструкция на " + distance;
    card.querySelector(".metrics").textContent =
      "Глобальный SSIM " + format(plane.ssim) + "\nPearson " + format(plane.pearson);
    const download = card.querySelector("a");
    download.href = plane.image;
    download.download = "restored_" + (index + 1) + ".png";
    grid.append(card);
  });
  result.hidden = false;
  document.querySelector("#result-title").focus();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy || inputs.some((item) => !item.data)) return;
  result.hidden = true;
  withStatus("Рассчитываю ДОЭ…", async () => {
    const data = await request("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ images: inputs.map((item) => item.data) }),
    });
    render(data);
    return "Расчёт завершён";
  });
});
