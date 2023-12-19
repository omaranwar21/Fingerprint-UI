window.onscroll = function () { stickyNavbar() };

var navbar = document.getElementById("navbar");
var sticky = navbar.offsetTop;

function stickyNavbar() {
  if (window.pageYOffset >= sticky) {
    navbar.classList.add("sticky");
  } else {
    navbar.classList.remove("sticky");
  }
}

// --------------------------------------------------- Prediction -----------------------------------------------------
// function showPreviewAndProcess(event) {
//   if (event.target.files.length > 0) {
//     var src = URL.createObjectURL(event.target.files[0]);
//     var preview = document.getElementById("file-ip-1-preview");
//     preview.src = src;
//     preview.style.display = "block";
//   }
// }

function showPreview(event) {
  if (event.target.files.length > 0) {
    var src = URL.createObjectURL(event.target.files[0]);
    var preview = document.getElementById("file-ip-1-preview");
    preview.src = src;
    preview.style.display = "block";

    // Initiate image processing on the server
    processImage(event.target.files[0]);
  }
}

function processImage(file) {
  var formData = new FormData();
  formData.append('image', file);

  console.log("-----------");
  fetch('/processing', {
    method: 'POST',
    body: formData
  }).then(response => response.blob())
            .then(blob => {
                // Create a new image element
                var img = document.createElement('img');
                // Set the source of the image to the received blob
                img.src = URL.createObjectURL(blob);

                // Append the image to the resultImageContainer
                document.getElementById('result-container').innerHTML = '';
                document.getElementById('result-container').appendChild(img);
            })
            .catch(error => {
                console.error('Error:', error);
            });

//  var result = document.getElementById("result");
//  result.src = "../Application_enhance.jpg";
//    result.src = ;
  // .then(response => response.json())
  // .then(data => {
  //     // Update the HTML elements with the processed information
  //     document.getElementById("gender").innerText = data.gender;
  //     document.getElementById("rightLeft").innerText = data.rightLeft;
  //     document.getElementById("fingerName").innerText = data.fingerName;
  // })
  // .catch(error => {
  //     console.error('Error:', error);
  // });
}