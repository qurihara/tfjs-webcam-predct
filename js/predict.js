eruda.init();
var modelname = './sign_language_vgg16/';
var size_x = 100;
var size_y = 100;

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
	modelname = 'https://qurihara.github.io/tfjs-webcam-predct/sign_language_vgg16/';
	size_x = 100;
	size_y = 100;
	loadModel(tf.loadLayersModel);
	loadDict();
	startWebcam();
  setInterval(predict, 1000);
});

$("#start-button2").click(function(){
	modelname = 'https://qurihara.github.io/tfjs-webcam-predct/mobnet2-flowers/';
	size_x = 224;
	size_y = 224;
	loadModel(tf.loadGraphModel);
	loadDict();
	startWebcam();
  setInterval(predict, 1000);
});

$("#start-button3").click(function(){
	modelname = 'https://qurihara.github.io/tfjs-webcam-predct/mobnet2-catdog/';
	size_x = 224;
	size_y = 224;
	loadModel(tf.loadGraphModel);
	loadDict();
	startWebcam();
  setInterval(predict, 1000);
});

//-----------------------
// load model
//-----------------------

let model;
async function loadModel(loadf) {
    let modelfile = modelname + 'model.json';
	console.log("model loading.. : " + modelfile);
	$("#console").html(`<li>model loading...</li>`);
	// model=await tf.loadModel(modelfile);
// 	model=await tf.loadLayersModel(modelfile);
// 	model=await tf.loadGraphModel(modelfile);
	model=await loadf(modelfile);
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


	// Older browsers might not implement mediaDevices at all, so we set an empty object first
	if (navigator.mediaDevices === undefined) {
		navigator.mediaDevices = {};
	}

	// Some browsers partially implement mediaDevices. We can't just assign an object
	// with getUserMedia as it would overwrite existing properties.
	// Here, we will just add the getUserMedia property if it's missing.
	if (navigator.mediaDevices.getUserMedia === undefined) {
		navigator.mediaDevices.getUserMedia = function(constraints) {

			// First get ahold of the legacy getUserMedia, if present
			var getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

			// Some browsers just don't implement it - return a rejected promise with an error
			// to keep a consistent interface
			if (!getUserMedia) {
				return Promise.reject(new Error('getUserMedia is not implemented in this browser'));
			}

			// Otherwise, wrap the call to the old navigator.getUserMedia with a Promise
			return new Promise(function(resolve, reject) {
				getUserMedia.call(navigator, constraints, resolve, reject);
			});
		}
	}

	navigator.mediaDevices.getUserMedia({ audio: false, video: true })
	.then(function(stream) {
		var video = document.querySelector('video');
		// Older browsers may not have srcObject
		if ("srcObject" in video) {
			video.srcObject = stream;
		} else {
			// Avoid using this in new browsers, as it is going away.
			video.src = window.URL.createObjectURL(stream);
		}
		video.onloadedmetadata = function(e) {
			video.play();
		};
	})
	.catch(function(err) {
		console.log(err.name + ": " + err.message);
	});

	// vendorUrl = window.URL || window.webkitURL;
	//
	// navigator.getMedia = navigator.getUserMedia ||
	// 					 navigator.webkitGetUserMedia ||
	// 					 navigator.mozGetUserMedia ||
	// 					 navigator.msGetUserMedia;
	//
	// navigator.getMedia({
	// 	video: true,
	// 	audio: false
	// }, function(stream) {
	// 	localStream = stream;
	// 	video.srcObject = stream;
	// 	video.play();
	// }, function(error) {
	// 	alert("Something wrong with webcam!");
	// });
}

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
	}).slice(0,classNames.length);//5);

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
	// let tensor = tf.fromPixels(image).resizeNearestNeighbor([100,100]).toFloat();
	let tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([size_x,size_y]).toFloat();
	let offset = tf.scalar(255);
    return tensor.div(offset).expandDims();
}

//-----------------------
// clear button event
//-----------------------

$("#clear-button").click(function clear() {
	location.reload();
});
