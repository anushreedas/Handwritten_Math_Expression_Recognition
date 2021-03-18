<!-- New endpoint from Alberto Serrano-Calva -->
var endpointUrl = "http://localhost:5050/api/predict"
var drawLaTeX;

<!-- Author: Michael Borcherds, Jan 2021 (minor edits by R. Zanibbi)

function addTextToDom(str) {
  // create a new div element
  const newDiv = document.createElement("div");

  // and give it some content
  const newContent = document.createTextNode(str);

  // add the text node to the newly created div
  newDiv.appendChild(newContent);

  // add the newly created element and its content into the DOM
  const currentDiv = document.getElementById("div1");
  document.body.insertBefore(newDiv, currentDiv);
}


function send(request) {
	//hand.setStrokes(penData);

    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
	console.log(xmlHttp);
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
            console.log("response = " + xmlHttp.responseText);
			console.log(xmlHttp.responseText.length);
			
			var response = JSON.parse(xmlHttp.responseText);
			console.log(response);
			console.log(response.mathml);
			console.log(response.latex);
			
			addTextToDom(response.latex);
			
			//alert(response.latex);
			if (drawLaTeX) {
				drawLaTeX(response.latex);
			}
			//var mathml = response.result.mathml;
			//var latex = MathML2LaTeX.convert(mathml);
			// remove strange thin space \u2062 that MathML2LaTeX inserts
			//latex = latex.replace(/\u2062/g,"");
			//console.log(latex);

			
    	} else {
			// WARNING: This is for debugging, removed to improve usability, not sure of other effects.
			//alert("xmlHttp.readyState = " + xmlHttp.readyState + "\nxmlHttp.status = " + xmlHttp.status + "\nxmlHttp.responseText = " + xmlHttp.responseText);
		}
	}
    xmlHttp.open("POST", endpointUrl, true); // true for asynchronous 
	xmlHttp.setRequestHeader('Content-Type', 'application/json');
    xmlHttp.send(JSON.stringify(request));
}

function wrapStrokes(strokes) {
return {
    "request_time": "2021-01-22T01:59:06",
    "input_type": "strokes",
    "file_id": "ISICal19_1201_em_751",
    //"annotation": "string",
    "input_strokes": strokes
	};


}

function sendStrokes() {

	var strokes = [];

	for (var id = 0 ; id < xcoords.length ; id ++) {
		var stroke = "";
		var xc = xcoords[id];
		var yc = ycoords[id];


		for (var i = 0 ; i < xc.length ; i++) {
			stroke += xc[i] + " " + yc[i] + (i==xc.length-1 ? "" : ",");
		}
		strokes[id] = {"points" : stroke, "id":id };
	}
	console.log(strokes);
	send(wrapStrokes(strokes));
}

function wrapImage(canvas) {
	var dataURL = canvas.toDataURL();
	console.log(dataURL);
	return wrapImageDataURL(dataURL);
}

function wrapImageDataURL(dataURL) {


return {
    "request_time": "2021-01-22T02:59:43",
    "input_type": "image",
    "file_id": "2800022",
    //"annotation": "string",
    "input_image": {
        "base64_img": dataURL.replace("data:image/png;base64,", ""),
        "dpi": 600
    }
}

}
