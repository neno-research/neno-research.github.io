<!DOCTYPE html>
<html>
<head>
  <title>HDC Playground: Multimodal Neural Net (JS)</title>
  <style>
    body { font-family: sans-serif; background: #f9fbfd; margin: 2em; }
    .container { display: flex; gap: 2em; }
    .column { flex: 1; min-width: 350px; }
    .section-title { font-weight: bold; margin-top: 1em; }
    label { display: block; margin-top: 0.5em; }
    input, select, textarea { margin-top: 0.2em; width: 97%; }
    button { margin: 0.3em 0.2em 0.3em 0; }
    .img-preview { margin: 0.5em 0; }
    table { border-collapse: collapse; margin-top: 0.5em; }
    td, th { border: 1px solid #ccc; padding: 4px 8px; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<h1>Hyperdimensional Neural Network Playground</h1>
<div class="container">
  <div class="column">

    <!-- Multimodal Single Input -->
    <div class="section-title">Single Input Prediction</div>
    <form id="singleForm">
      <label>Color:
        <select id="colorSel">
          <option>red</option><option>green</option><option>blue</option>
        </select>
      </label>
      <label>Shape:
        <select id="shapeSel">
          <option>circle</option><option>square</option><option>triangle</option>
        </select>
      </label>
      <label>Length: <input type="number" step="0.1" id="lengthInp" min="0" max="10" value="5"/></label>
      <label>Width: <input type="number" step="0.1" id="widthInp" min="0" max="10" value="5"/></label>
      <label>Text (n-gram): <input type="text" id="textInp" placeholder="e.g. fast bright sun"/></label>
      <label>Upload Tiny Image: <input type="file" id="imgInp" accept="image/*"></label>
      <canvas id="imgPrev" width="16" height="16" style="display:none;" class="img-preview"></canvas>
      <button type="submit">Predict</button>
    </form>
    <button id="learnBtn">Learn from This Example</button>
    <div id="singleResult"></div>

    <!-- Batch and Confusion Matrix -->
    <div class="section-title">Batch Upload (CSV: color,shape,length,width,text,label)</div>
    <textarea id="csvInput" rows="5" placeholder="red,circle,9.1,3.2,fast cat,hot"></textarea><br>
    <button id="batchBtn">Batch Predict</button>
    <button id="clearBtn">Clear Results</button>
    <div id="batchResult"></div>
    <div>
      <canvas id="simChart" height="170"></canvas>
      <canvas id="confChart" height="170"></canvas>
    </div>

    <!-- Export/Import -->
    <div class="section-title">Export/Import Model</div>
    <button id="exportBtn">Export Model</button>
    <input type="file" id="importInp" style="display:none;">
    <button id="importBtn">Import Model</button>
    <div id="exportStatus"></div>
  </div>
  <div class="column">
    <div class="section-title">Hypervector Space (PCA)</div>
    <canvas id="pcaChart" width="380" height="260"></canvas>
    <div style="font-size:0.8em; color:#888;">Projection: memories &amp; inputs. Hover for details.</div>
  </div>
</div>
<script>
/* ======= HDC core ======= */
function randomHV(dim) {
  return Array.from({length: dim}, () => Math.random() > 0.5 ? 1 : -1);
}
function bind(a, b) { return a.map((v, i) => v * b[i]); }
function superpose(vecs) {
  let sum = Array(vecs[0].length).fill(0);
  for (const v of vecs) for (let i=0; i<v.length; i++) sum[i] += v[i];
  return sum.map(x => x >= 0 ? 1 : -1);
}
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for(let i = 0; i < a.length; i++) { dot += a[i]*b[i]; normA += a[i]*a[i]; normB += b[i]*b[i]; }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
function quantize(val, nBins, min, max) {
  return Math.min(nBins - 1, Math.max(0, Math.floor((val - min) / (max - min + 1e-9) * nBins)));
}
// N-gram encode
function encodeNgram(words, hvDict, posDict, dim) {
  let hvs = [];
  for (let i = 0; i < words.length; i++) {
    let key = 'ng:' + words[i];
    if (!hvDict[key]) hvDict[key] = randomHV(dim);
    if (!posDict[i]) posDict[i] = randomHV(dim);
    hvs.push(bind(hvDict[key], posDict[i]));
  }
  return hvs.length ? superpose(hvs) : Array(dim).fill(0);
}
// Image encode: grayscale, each px = bind(intensity, position)
function encodeImage(pixels, width, height, hvDict, posDict, dim, nBins=8) {
  let hvs = [];
  for (let i = 0; i < width * height; i++) {
    let bin = quantize(pixels[i], nBins, 0, 255);
    let intKey = 'px:' + bin;
    if (!hvDict[intKey]) hvDict[intKey] = randomHV(dim);
    if (!posDict[i]) posDict[i] = randomHV(dim);
    hvs.push(bind(hvDict[intKey], posDict[i]));
  }
  return hvs.length ? superpose(hvs) : Array(dim).fill(0);
}
function encodeFeatures(features, featureSpaces, hvDict, posDict, dim) {
  // Compose feature HVs: categorical, numeric, ngram, image
  let hvList = [];
  for (const [name, val] of Object.entries(features)) {
    const meta = featureSpaces[name];
    if (!meta) continue;
    let key;
    if (meta.type === "cat") {
      key = name + ':' + val;
      if (!hvDict[key]) hvDict[key] = randomHV(dim);
      hvList.push(hvDict[key]);
    } else if (meta.type === "num") {
      let bin = quantize(val, meta.nBins, meta.min, meta.max);
      key = name + ':bin:' + bin;
      if (!hvDict[key]) hvDict[key] = randomHV(dim);
      hvList.push(hvDict[key]);
    } else if (meta.type === "ngram") {
      hvList.push(encodeNgram(val, hvDict, posDict, dim));
    } else if (meta.type === "image") {
      hvList.push(encodeImage(val.pixels, val.width, val.height, hvDict, posDict, dim));
    }
  }
  return hvList.length ? superpose(hvList) : Array(dim).fill(0);
}
class HDCClassifier {
  constructor(classLabels, dim=2000) {
    this.classLabels = classLabels;
    this.dim = dim;
    this.classMemory = {}; this.classLabels.forEach(c => { this.classMemory[c] = Array(dim).fill(0); });
    this.hvDict = {}; this.posDict = {}; this.featureSpaces = {};
  }
  setFeatureSpaces(fs) { this.featureSpaces = fs; }
  train(features, label) {
    let enc = encodeFeatures(features, this.featureSpaces, this.hvDict, this.posDict, this.dim);
    for (let i=0; i<this.dim; i++) this.classMemory[label][i] += enc[i];
  }
  predict(features) {
    let enc = encodeFeatures(features, this.featureSpaces, this.hvDict, this.posDict, this.dim);
    let scores = {};
    for (const c of this.classLabels)
      scores[c] = cosineSimilarity(enc, this.classMemory[c]);
    let pred = Object.keys(scores).reduce((a, b) => scores[a] > scores[b] ? a : b);
    return {pred, scores, enc};
  }
  export() {
    return JSON.stringify({
      classLabels: this.classLabels,
      dim: this.dim,
      classMemory: this.classMemory,
      hvDict: this.hvDict,
      posDict: this.posDict
    });
  }
  import(json) {
    let data = typeof json === "string" ? JSON.parse(json) : json;
    this.classLabels = data.classLabels;
    this.dim = data.dim;
    this.classMemory = data.classMemory;
    this.hvDict = data.hvDict;
    this.posDict = data.posDict;
  }
}

/* ======= Model Setup ======= */
const DIM = 2000;
const CLASSES = ["hot", "warm", "cold"];
const clf = new HDCClassifier(CLASSES, DIM);
clf.setFeatureSpaces({
  color: {type: "cat", values: ["red", "green", "blue"]},
  shape: {type: "cat", values: ["circle", "square", "triangle"]},
  length: {type: "num", min: 0, max: 10, nBins: 10},
  width: {type: "num", min: 0, max: 10, nBins: 10},
  ngram: {type: "ngram"},
  image: {type: "image"}
});
let trainData = [
  {features: {color: "red", shape: "circle", length: 8, width: 6, ngram: ["fast","bright"], image: {pixels:Array(16).fill(240),width:4,height:4}}, label: "hot"},
  {features: {color: "green", shape: "triangle", length: 2, width: 2, ngram: ["slow","dull"], image: {pixels:Array(16).fill(50),width:4,height:4}}, label: "cold"},
  {features: {color: "blue", shape: "square", length: 4, width: 6, ngram: ["neutral"], image: {pixels:Array(16).fill(150),width:4,height:4}}, label: "warm"},
  {features: {color: "red", shape: "triangle", length: 9, width: 9, ngram: ["hot"], image: {pixels:Array(16).fill(250),width:4,height:4}}, label: "hot"},
  {features: {color: "green", shape: "circle", length: 1, width: 2, ngram: ["cold"], image: {pixels:Array(16).fill(20),width:4,height:4}}, label: "cold"}
];
trainData.forEach(ex => clf.train(ex.features, ex.label));

/* ======= UI Logic ======= */
const colorSel = document.getElementById('colorSel');
const shapeSel = document.getElementById('shapeSel');
const lengthInp = document.getElementById('lengthInp');
const widthInp = document.getElementById('widthInp');
const textInp = document.getElementById('textInp');
const imgInp = document.getElementById('imgInp');
const imgPrev = document.getElementById('imgPrev');
const singleForm = document.getElementById('singleForm');
const singleResult = document.getElementById('singleResult');
const learnBtn = document.getElementById('learnBtn');
const csvInput = document.getElementById('csvInput');
const batchBtn = document.getElementById('batchBtn');
const clearBtn = document.getElementById('clearBtn');
const batchResult = document.getElementById('batchResult');
let simChart = null, confChart = null, pcaChart = null, pcaVectors = [];
let lastSingleFeatures = null, lastSinglePred = null;
let tinyImage = {pixels:Array(16).fill(150),width:4,height:4}; // default

imgInp.onchange = (e) => {
  if (!e.target.files[0]) return;
  let fr = new FileReader();
  fr.onload = function(ev) {
    let img = new window.Image();
    img.onload = function() {
      let cv = imgPrev, ctx = cv.getContext('2d');
      ctx.clearRect(0,0,cv.width,cv.height);
      ctx.drawImage(img, 0, 0, 4, 4);
      let data = ctx.getImageData(0, 0, 4, 4).data;
      let px = [];
      for (let i=0; i<data.length; i+=4)
        px.push(Math.round((data[i]+data[i+1]+data[i+2])/3));
      tinyImage = {pixels:px, width:4, height:4};
      imgPrev.style.display = '';
    };
    img.src = ev.target.result;
  };
  fr.readAsDataURL(e.target.files[0]);
};

singleForm.onsubmit = (e) => {
  e.preventDefault();
  let color = colorSel.value, shape = shapeSel.value, length = parseFloat(lengthInp.value), width = parseFloat(widthInp.value);
  let ngram = textInp.value.trim().split(/\s+/).filter(Boolean);
  let features = {color, shape, length, width, ngram, image: tinyImage};
  let result = clf.predict(features);
  let colormap = {hot:"#e1496a", cold:"#2a91d9", warm:"#faad15"};
  singleResult.innerHTML =
    `<b>Prediction:</b> <span style="color:${colormap[result.pred]}">${result.pred}</span><br>` +
    CLASSES.map(c => `&nbsp;&nbsp;${c}: ${result.scores[c].toFixed(3)}`).join("<br>");
  lastSingleFeatures = features;
  lastSinglePred = result.pred;
  // PCA update
  updatePCAVectors([{vec:result.enc,label:"input",type:"test",idx:0}]);
  updatePCAChart();
  updateSimChart([{scores:result.scores}]);
};

learnBtn.onclick = () => {
  let label = prompt(`Enter the true label for this example (${CLASSES.join(", ")}):`);
  if (label && CLASSES.includes(label) && lastSingleFeatures) {
    clf.train(lastSingleFeatures, label);
    alert("Learned! The model has been updated.");
  }
};

batchBtn.onclick = () => {
  let lines = csvInput.value.trim().split("\n").map(l=>l.trim()).filter(x=>x);
  let batchResults = [], trueLabels = [], predLabels = [];
  let pcaVecs = [];
  // Add class memories to PCA
  for (let c of CLASSES) pcaVecs.push({vec: clf.classMemory[c].slice(), label: c, type:"memory", idx:0});
  let out = "";
  lines.forEach((line,i) => {
    let [color,shape,length,width,text,label] = line.split(",");
    if (!color || !shape || !length || !width) return;
    let ngram = text?text.trim().split(/\s+/).filter(Boolean):[];
    let features = {color: color.trim(), shape: shape.trim(), length: parseFloat(length), width: parseFloat(width), ngram, image: tinyImage};
    let res = clf.predict(features);
    batchResults.push({scores: res.scores, label, features, enc: res.enc, pred: res.pred});
    trueLabels.push(label);
    predLabels.push(res.pred);
    pcaVecs.push({vec: res.enc.slice(), label: (label || "test"), type:"test", idx: i+1});
    out += `Line ${i+1}: True <b>${label||"?"}</b>, Pred <b style="color:${res.pred===label?'#19a33d':'#e1496a'}">${res.pred}</b>, `
        + CLASSES.map(c => `${c}: ${res.scores[c].toFixed(3)}`).join(", ") + "<br>";
  });
  batchResult.innerHTML = out;
  updateSimChart(batchResults);
  updateConfChart(trueLabels, predLabels);
  updatePCAVectors(pcaVecs);
  updatePCAChart();
};

clearBtn.onclick = () => {
  batchResult.innerHTML = "";
  if (simChart) simChart.destroy();
  if (confChart) confChart.destroy();
};

function updateSimChart(batchResults) {
  const labels = batchResults.map((r, i) => "Test " + (i+1));
  let colors = ["#e1496a","#faad15","#2a91d9"];
  let datasets = CLASSES.map((c,ci)=>({
    label: c.charAt(0).toUpperCase()+c.slice(1),
    data: batchResults.map(r=>r.scores[c]),
    backgroundColor: colors[ci]+"80"
  }));
  if (simChart) simChart.destroy();
  const ctx = document.getElementById('simChart').getContext('2d');
  simChart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: { plugins: { legend: { display: true }},
      scales: { y: { beginAtZero: true, min: -1, max: 1, title: {display:true,text:"Cosine Similarity"}} }
    }
  });
}

function updateConfChart(trueLabels, predLabels) {
  // Confusion matrix
  let cm = {}, n = CLASSES.length;
  for (let t of CLASSES) cm[t] = {}; for (let t of CLASSES) for (let p of CLASSES) cm[t][p]=0;
  for (let i=0; i<trueLabels.length; i++)
    if (CLASSES.includes(trueLabels[i]) && CLASSES.includes(predLabels[i]))
      cm[trueLabels[i]][predLabels[i]]++;
  // Accuracy
  let correct = 0, total = 0;
  for (let i=0; i<trueLabels.length; i++)
    if (trueLabels[i] && predLabels[i]) { total++; if (trueLabels[i]===predLabels[i]) correct++; }
  let accuracy = total? (correct/total*100).toFixed(1)+"%":"-";
  // Chart
  let data = [];
  for (let t of CLASSES)
    data.push({label:t, data:CLASSES.map(p=>cm[t][p]), backgroundColor:"#aaa"});
  if (confChart) confChart.destroy();
  let ctx = document.getElementById('confChart').getContext('2d');
  confChart = new Chart(ctx, {
    type: 'bar',
    data: { labels: CLASSES, datasets: data },
    options: {
      plugins: { legend:{display:false}, title:{display:true, text:'Confusion Matrix (Accuracy: '+accuracy+')'}},
      scales: { x:{title:{display:true,text:"Predicted"}}, y:{title:{display:true,text:"Count"}, beginAtZero:true } }
    }
  });
}
function updatePCAVectors(vecs) { pcaVectors = []; if (vecs) pcaVectors=vecs; }
function pca2d(matrix) {
  let mean = Array(matrix[0].length).fill(0);
  for (let row of matrix) for (let i=0; i<row.length; i++) mean[i] += row[i];
  mean = mean.map(x => x/matrix.length);
  let X = matrix.map(row => row.map((v,i) => v-mean[i]));
  let cov = Array(X[0].length).fill(0).map(()=>Array(X[0].length).fill(0));
  for(let i=0; i<X[0].length; i++) for(let j=0; j<X[0].length; j++) for(let r of X) cov[i][j] += r[i]*r[j];
  for(let i=0;i<X[0].length;i++) for(let j=0;j<X[0].length;j++) cov[i][j]/=X.length;
  function powerIter(mat, nIter=20) {
    let v = Array(mat.length).fill(0).map(()=>Math.random());
    for(let iter=0; iter<nIter; iter++) {
      let v1 = Array(mat.length).fill(0);
      for(let i=0;i<mat.length;i++) for(let j=0;j<mat.length;j++) v1[i]+=mat[i][j]*v[j];
      let norm = Math.sqrt(v1.reduce((s,x)=>s+x*x,0));
      v = v1.map(x=>x/norm);
    }
    return v;
  }
  let pc1 = powerIter(cov);
  let lambda1 = 0;
  for(let i=0;i<cov.length;i++) for(let j=0;j<cov.length;j++) lambda1+=pc1[i]*cov[i][j]*pc1[j];
  let cov2 = cov.map((row,i)=>row.map((x,j)=>x-lambda1*pc1[i]*pc1[j]));
  let pc2 = powerIter(cov2);
  let proj = X.map(row=>[row.reduce((s,v,i)=>s+v*pc1[i],0), row.reduce((s,v,i)=>s+v*pc2[i],0)]);
  return proj;
}
function updatePCAChart() {
  if (pcaVectors.length < 2) return;
  let allVecs = pcaVectors.map(v=>v.vec);
  let proj = pca2d(allVecs);
  let colors = {hot:"#e1496a", cold:"#2a91d9", warm:"#faad15", memory:"#333"};
  if (pcaChart) pcaChart.destroy();
  let ctx = document.getElementById('pcaChart').getContext('2d');
  let datasets = [];
  for (let type of ["memory","test"]) {
    let data = proj.map((p,i)=>pcaVectors[i].type===type ? {x:p[0],y:p[1],label:pcaVectors[i].label,idx:pcaVectors[i].idx}:null).filter(x=>x);
    datasets.push({
      label: type==="memory" ? "Class Memory" : "Input",
      data: data,
      pointRadius: type==="memory"?7:5,
      backgroundColor: type==="memory"?"#333":(x=>colors[x.label]||"#5bb6f9")
    });
  }
  pcaChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      plugins: { legend:{display:true}, tooltip:{callbacks:{
        label:function(context){return context.raw.label+" ("+context.dataset.label+")";}
      }}},
      scales: {x:{title:{display:true,text:"PCA-1"}}, y:{title:{display:true,text:"PCA-2"}}}
    }
  });
}

/* ===== Export/Import Model ===== */
const exportBtn = document.getElementById('exportBtn');
const importBtn = document.getElementById('importBtn');
const importInp = document.getElementById('importInp');
const exportStatus = document.getElementById('exportStatus');
exportBtn.onclick = () => {
  let json = clf.export();
  let blob = new Blob([json], {type: "application/json"});
  let a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "hdc_model.json";
  document.body.appendChild(a); a.click(); a.remove();
  exportStatus.innerHTML = "Model exported!";
};
importBtn.onclick = () => { importInp.click(); };
importInp.onchange = (e) => {
  if (!e.target.files[0]) return;
  let fr = new FileReader();
  fr.onload = function(ev) {
    clf.import(ev.target.result);
    exportStatus.innerHTML = "Model imported!";
  };
  fr.readAsText(e.target.files[0]);
};

/* ===== Example Batch ===== */
csvInput.value =
`red,circle,9.1,3.1,fast bright sun,hot
blue,square,3.5,5.5,neutral,warm
green,triangle,1.5,1.2,slow dull,cold
red,triangle,8.9,8.8,hot,hot
green,circle,2.2,3.2,cold,cold`;

/* ===== Init: Show class memories on PCA ===== */
updatePCAVectors(CLASSES.map(c=>({vec:clf.classMemory[c],label:c,type:"memory",idx:0})));
updatePCAChart();
</script>
</body>
</html>
