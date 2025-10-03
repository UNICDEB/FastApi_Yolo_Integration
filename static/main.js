function startCamera() {
  fetch("/start-camera/", { method: "POST" })
    .then(response => response.json())
    .then(data => alert(data.status))
    .catch(err => alert("Error starting camera"));
}

function showSpinner() {
  document.getElementById("spinner").style.display = "block";
}

function clearResults() {
  // Hide spinner if visible
  document.getElementById("spinner").style.display = "none";
  // Remove results
  const resultDiv = document.getElementById("results");
  if (resultDiv) {
    resultDiv.innerHTML = "";
  }
}