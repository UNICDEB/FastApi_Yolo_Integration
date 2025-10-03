// function startCamera() {
//   fetch("/start-camera/", { method: "POST" })
//     .then(response => response.json())
//     .then(data => alert(data.status))
//     .catch(err => alert("Error starting camera"));
// }

// function showSpinner() {
//   document.getElementById("spinner").style.display = "block";
// }

// function clearResults() {
//   // Hide spinner if visible
//   document.getElementById("spinner").style.display = "none";
//   // Remove results
//   const resultDiv = document.getElementById("results");
//   if (resultDiv) {
//     resultDiv.innerHTML = "";
//   }
// }

function startCamera() {
  fetch("/start-camera/", { method: "POST" })
    .then(response => response.json())
    .then(data => {
      alert(data.status);
      if (data.status === "Camera started") {
        const v = document.getElementById('video_stream');
        if (v) v.style.display = 'block';
      }
    })
    .catch(err => alert("Error starting camera"));
}

function stopCamera() {
  fetch("/stop-camera/", { method: "POST" })
    .then(response => response.json())
    .then(data => {
      alert(data.status);
      const v = document.getElementById('video_stream');
      if (v) v.style.display = 'none';
    })
    .catch(err => alert("Error stopping camera"));
}

function showSpinner() {
  document.getElementById("spinner").style.display = "block";
}

function clearResults() {
  // Hide spinner if visible
  const spinnerDiv = document.getElementById("spinner");
  if (spinnerDiv) spinnerDiv.style.display = "none";

  // Remove results area content
  const resultDiv = document.getElementById("results");
  if (resultDiv) {
    resultDiv.innerHTML = "";
  }
}
