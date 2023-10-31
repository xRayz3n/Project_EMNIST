const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function draw(e) {
    if (!isDrawing) return;
    ctx.lineWidth = 10;  // Change as needed
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

document.getElementById('clear').addEventListener('click', function() {
    let ctx = canvas.getContext('2d');
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
});
document.getElementById('predict').addEventListener('click', function() {
    predict();
});
async function predict() {
    // Redimensionnez l'image du canvas à 28x28
    let tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = 28;
    tmpCanvas.height = 28;
    let tmpCtx = tmpCanvas.getContext('2d');
  
    tmpCtx.drawImage(canvas, 0, 0, 28, 28);
  
  
    // Extraire les pixels et les convertir en niveaux de gris
    let NOT_ZERO_NUMB = 0
    let pix_num = 0
    let imgData = tmpCtx.getImageData(0, 0, 28, 28).data;
  
  
    for (let i = 0; i < imgData.length; i+=1){
        if (imgData[i]!=0){
            NOT_ZERO_NUMB += 1; }
        pix_num += 1
    }
  
    let input = new Float32Array(28 * 28);
    //let array2D = [];
    for (let i = 0; i < imgData.length; i += 4) {
        let grayscale = 0
        if (imgData[i]!=0)  {grayscale = 255}
        if (imgData[i+1]!=0)  {grayscale = 255}
        if (imgData[i+2]!=0)  {grayscale = 255}
        grayscale = (((grayscale/ 255)) - 0.1736) / 0.3317;
        // Normaliser entre -1 et 1
        input[i/4] = grayscale
    }
  
    // Binariser l'image
    for (let i = 0; i < input.length; i += 1) {
      if (input[i] < 1) {
        input[i] = 0;
      }
    }
    let transposedInput = new Float32Array(28 * 28);
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            transposedInput[x * 28 + y] = input[y * 28 + x];
        }
    }

    let tensorInput = new onnx.Tensor(transposedInput, 'float32', [1, 1, 28, 28]);

    let outputMap = await model.run([tensorInput]);
    
    let outputData = outputMap.values().next().value.data;
  
    // Retourner la classe avec la plus haute probabilité
    let pred_class = outputData.indexOf(Math.max(...outputData))
    let pred_char = String.fromCharCode(pred_class + 64)
    console.log("Class: " + pred_char)
    document.getElementById('result').innerText = `Predicted letter: ${pred_char}`;
    return pred_class;
  }

  async function loadModel() {
    model = new onnx.InferenceSession();
    await model.loadModel("emnist.onnx");
}
let model;
loadModel()