function predict() {
    let formData = new FormData(document.getElementById('predict-form'));

    fetch('/predict', {
        method: 'POST',
        body: new URLSearchParams(formData)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = "Disease: " + data.disease + ", Severity: " + data.severity;
    })
    .catch(error => console.error('Error:', error));
}
