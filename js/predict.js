eruda.init();
let modelname = './sign_language_vgg16/';
//let CLASSES = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four',5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine'}

var classNames = [];
/*
load the class names
*/
async function loadDict() {
    loc = modelname + 'class_names.txt'
    await $.ajax({
        url: loc,
        dataType: 'text',
    }).done(success);
}
/*
load the class names
*/
function success(data) {
    const lst = data.split(/\n/)
    for (var i = 0; i < lst.length - 1; i++) {
        let symbol = lst[i]
        classNames[i] = symbol
    }
}

// const names = getClassNames(indices);
// function getClassNames(indices) {
//     var outp = []
//     for (var i = 0; i < indices.length; i++)
//         outp[i] = classNames[indices[i]]
//     return outp
// }


//-----------------------
// start button event
//-----------------------

$("#start-button").click(function(){
	loadModel();
	loadDict();
	startWebcam();
});

//-----------------------
// load model
//-----------------------

let modelfile = modelname + 'model.json';
let model;
async function loadModel() {
	console.log("model loading..");
	$("#console").html(`<li>model loading...</li>`);
	model=await tf.loadModel(modelfile);
	console.log("model loaded.");
	$("#console").html('<li>' + modelname + ' loaded.</li>');
};

//-----------------------
// start webcam
//-----------------------

var video;
function startWebcam() {
	console.log("video streaming start.");
	$("#console").html(`<li>video streaming start.</li>`);
	video = $('#main-stream-video').get(0);
	vendorUrl = window.URL || window.webkitURL;

	navigator.getMedia = navigator.getUserMedia ||
						 navigator.webkitGetUserMedia ||
						 navigator.mozGetUserMedia ||
						 navigator.msGetUserMedia;

	navigator.getMedia({
		video: true,
		audio: false
	}, function(stream) {
		localStream = stream;
		video.srcObject = stream;
		video.play();
	}, function(error) {
		alert("Something wrong with webcam!");
	});
}

//-----------------------
// predict button event
//-----------------------

$("#predict-button").click(function(){
	setInterval(predict, 1000/10);
});

//-----------------------
// TensorFlow.js method
// predict tensor
//-----------------------

async function predict(){
	let tensor = captureWebcam();

	let prediction = await model.predict(tensor).data();
	let results = Array.from(prediction)
				.map(function(p,i){
	return {
		probability: p,
		className: classNames[i]
	};
	}).sort(function(a,b){
		return b.probability-a.probability;
	}).slice(0,5);

	$("#console").empty();

	results.forEach(function(p){
		$("#console").append(`<li>${p.className} : ${p.probability.toFixed(6)}</li>`);
		console.log(p.className,p.probability.toFixed(6))
	});

};

//------------------------------
// capture streaming video
// to a canvas object
//------------------------------

function captureWebcam() {
	var canvas    = document.createElement("canvas");
	var context   = canvas.getContext('2d');
	canvas.width  = video.width;
	canvas.height = video.height;

	context.drawImage(video, 0, 0, video.width, video.height);
	tensor_image = preprocessImage(canvas);

	return tensor_image;
}

//-----------------------
// TensorFlow.js method
// image to tensor
//-----------------------

function preprocessImage(image){
	let tensor = tf.fromPixels(image).resizeNearestNeighbor([100,100]).toFloat();
	let offset = tf.scalar(255);
    return tensor.div(offset).expandDims();
}

//-----------------------
// clear button event
//-----------------------

$("#clear-button").click(function clear() {
	location.reload();
});
