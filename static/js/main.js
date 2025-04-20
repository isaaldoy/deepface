// static/js/main.js

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('captureBtn');
const messageArea = document.getElementById('messageArea');
const usernameInput = document.getElementById('username'); // Only on register page

let currentMode = ''; // 'login' or 'register'

function displayMessage(message, type = 'info') {
    messageArea.textContent = message;
    messageArea.className = `message ${type}`; // Apply styling class
    messageArea.style.display = 'block';
}

function setupWebcam(mode) {
    currentMode = mode;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        displayMessage("Webcam access not supported by this browser.", "error");
        captureBtn.disabled = true;
        return;
    }

    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(stream => {
            video.srcObject = stream;
            video.play();
            captureBtn.disabled = false; // Enable button once webcam is ready
            displayMessage("Webcam ready. Position your face clearly and capture.", "info");
        })
        .catch(err => {
            console.error("Error accessing webcam:", err);
            displayMessage(`Error accessing webcam: ${err.name}. Please ensure permissions are granted.`, "error");
            captureBtn.disabled = true;
        });
}

captureBtn.addEventListener('click', () => {
    if (!video.srcObject) {
        displayMessage("Webcam not active.", "error");
        return;
    }

    // Draw current video frame to canvas
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get image data from canvas as JPEG base64
    const imageDataUrl = canvas.toDataURL('image/jpeg');

    // Disable button during processing
    captureBtn.disabled = true;
    captureBtn.textContent = 'Processing...';
    displayMessage("Processing image...", "info");

    if (currentMode === 'register') {
        handleRegistration(imageDataUrl);
    } else if (currentMode === 'login') {
        handleLogin(imageDataUrl);
    } else {
         displayMessage("Invalid mode.", "error");
         captureBtn.disabled = false;
         captureBtn.textContent = 'Capture Photo';
    }
});

function handleRegistration(imageDataUrl) {
    const username = usernameInput.value.trim();
    if (!username) {
        displayMessage("Please enter a username.", "error");
        captureBtn.disabled = false;
        captureBtn.textContent = 'Capture Registration Photo';
        return;
    }

    fetch('/register', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username: username, image: imageDataUrl }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayMessage(data.message, "success");
            // Redirect to login page after a short delay
            setTimeout(() => { window.location.href = '/login'; }, 2000);
        } else {
            displayMessage(data.message || "Registration failed.", "error");
        }
    })
    .catch(error => {
        console.error('Error during registration fetch:', error);
        displayMessage("An error occurred connecting to the server.", "error");
    })
    .finally(() => {
        // Re-enable button unless redirecting
        if (!document.querySelector('.message.success')) { // Check if success message is shown
            captureBtn.disabled = false;
            captureBtn.textContent = 'Capture Registration Photo';
        }
    });
}


function handleLogin(imageDataUrl) {
     fetch('/authenticate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageDataUrl }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayMessage(data.message, "success");
             // Redirect to home/profile page on successful login
            setTimeout(() => { window.location.href = '/'; }, 1500);
        } else {
            displayMessage(data.message || "Login failed.", "error");
             // Re-enable button on failure
            captureBtn.disabled = false;
            captureBtn.textContent = 'Capture Login Photo';
        }
    })
    .catch(error => {
        console.error('Error during authentication fetch:', error);
        displayMessage("An error occurred connecting to the server.", "error");
        captureBtn.disabled = false;
        captureBtn.textContent = 'Capture Login Photo';
    });
}

// Initial setup call might be needed if script loads before setupWebcam is called from template
// document.addEventListener('DOMContentLoaded', () => {
//    // Check which page we are on if setupWebcam isn't called directly
// });