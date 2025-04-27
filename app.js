// Audio recording variables
let audioContext;
let mediaRecorder;
let audioChunks = [];
let audioBlob = null;
let audioBuffer = null;
let visualizationInterval = null;
const apiUrl = 'https://8cb0-103-197-112-241.ngrok-free.app//audio_predict';

// DOM elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const statusDiv = document.getElementById('status');
const resultsDiv = document.getElementById('results');
const predictionResultDiv = document.getElementById('predictionResult');
const featureTableBody = document.getElementById('featureTableBody');
const loadingDiv = document.getElementById('loading');
const canvas = document.getElementById('audioVisualizer');
const uploadInput = document.getElementById('uploadInput');
const canvasCtx = canvas.getContext('2d');

// Set canvas dimensions
function resizeCanvas() {
    const container = document.querySelector('.visualization');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Handle file upload
uploadInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) {
        statusDiv.textContent = "No file selected.";
        statusDiv.style.backgroundColor = "#ffebee";
        return;
    }
    
    if (!file.type.startsWith('audio/')) {
        statusDiv.textContent = "Please upload an audio file (e.g., WAV, MP3).";
        statusDiv.style.backgroundColor = "#ffebee";
        uploadInput.value = '';
        return;
    }
    
    try {
        statusDiv.textContent = "Processing uploaded audio...";
        statusDiv.style.backgroundColor = "";
        
        audioChunks = [];
        audioBlob = null;
        resultsDiv.style.display = 'none';
        analyzeBtn.style.display = 'none';
        
        audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
        
        const arrayBuffer = await file.arrayBuffer();
        
        try {
            audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            audioBlob = new Blob([file], { type: file.type });
            
            if (file.type !== 'audio/wav' && file.type !== 'audio/wave') {
                console.log("Converting uploaded file to WAV...");
                const wavBuffer = encodeWAV(audioBuffer);
                audioBlob = new Blob([wavBuffer], { type: 'audio/wav' });
                console.log("Converted WAV blob size:", audioBlob.size);
            }
            
            console.log("Uploaded audio blob size:", audioBlob.size);
            console.log("Uploaded audio blob type:", audioBlob.type);
            
            const url = URL.createObjectURL(audioBlob);
            const audio = new Audio(url);
            audio.play().catch(e => console.error("Playback failed:", e));
            
            analyzeBtn.style.display = 'inline-block';
            statusDiv.textContent = "Audio uploaded. Click 'Analyze Recording' to process.";
            
            drawWaveform();
            
        } catch (e) {
            console.error("Failed to decode uploaded audio:", e);
            statusDiv.textContent = "Invalid audio file. Please upload a valid audio file.";
            statusDiv.style.backgroundColor = "#ffebee";
            uploadInput.value = '';
        }
        
    } catch (error) {
        console.error("Error processing uploaded file:", error);
        statusDiv.textContent = `Error processing file: ${error.message}`;
        statusDiv.style.backgroundColor = "#ffebee";
        uploadInput.value = '';
    }
});

// Start recording
startBtn.addEventListener('click', async () => {
    try {
        statusDiv.textContent = "Initializing microphone...";
        statusDiv.style.backgroundColor = "";

        audioChunks = [];
        audioBlob = null;
        resultsDiv.style.display = 'none';
        analyzeBtn.style.display = 'none';
        uploadInput.value = '';

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        audioContext = new (window.AudioContext || window.webkitAudioContext)();

        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const webmBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const arrayBuffer = await webmBlob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

            // Convert to WAV format
            const wavBuffer = encodeWAV(audioBuffer);
            audioBlob = new Blob([wavBuffer], { type: 'audio/wav' });

            console.log("Audio blob size:", audioBlob.size);
            console.log("Audio blob type:", audioBlob.type);

            const url = URL.createObjectURL(audioBlob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'recorded_audio.wav';
            a.click();
            URL.revokeObjectURL(url);

            const audio = new Audio(url);
            audio.play().catch(e => console.error("Playback failed:", e));

            analyzeBtn.style.display = 'inline-block';
            statusDiv.textContent = "Recording complete. Click 'Analyze Recording' to process.";

            try {
                drawWaveform();
            } catch (e) {
                console.error("Failed to decode audio for visualization:", e);
            }
        };

        mediaRecorder.start();
        startBtn.style.display = 'none';
        stopBtn.style.display = 'inline-block';
        statusDiv.textContent = "Recording... Speak now.";

        startVisualization(stream);

    } catch (error) {
        console.error("Error accessing microphone:", error);
        statusDiv.textContent = `Error accessing microphone: ${error.message}. Ensure a microphone is connected and permissions are granted.`;
        statusDiv.style.backgroundColor = "#ffebee";
    }
});

// Stop recording
stopBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();

        mediaRecorder.stream.getTracks().forEach(track => track.stop());

        stopVisualization();

        startBtn.style.display = 'inline-block';
        stopBtn.style.display = 'none';
    }
});

// Analyze recording
analyzeBtn.addEventListener('click', async () => {
    if (!audioBlob) {
        statusDiv.textContent = "No recording or uploaded audio available to analyze.";
        statusDiv.style.backgroundColor = "#ffebee";
        return;
    }
    
    if (audioBlob.size < 1000) {
        statusDiv.textContent = "Audio is too short or empty. Please record or upload a longer audio.";
        statusDiv.style.backgroundColor = "#ffebee";
        return;
    }
    
    try {
        loadingDiv.style.display = 'block';
        analyzeBtn.disabled = true;
        statusDiv.textContent = "Processing your voice recording...";
        statusDiv.style.backgroundColor = "";
        
        let wavBlob = audioBlob;
        if (audioBlob.type !== 'audio/wav' && audioBlob.type !== 'audio/wave') {
            console.log("Converting to WAV...");
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            const wavBuffer = encodeWAV(audioBuffer);
            wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
            console.log("Converted WAV blob size:", wavBlob.size);
        }
        
        const formData = new FormData();
        formData.append('audio', wavBlob, 'recording.wav');
        
        console.log("Sending audio to server:", {
            size: wavBlob.size,
            type: wavBlob.type
        });
        
        const response = await fetch(apiUrl, {
            method: 'POST',
            body: formData
        });
        
        console.log("Server response status:", response.status);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.error || 'Unknown error'}`);
        }
        
        const data = await response.json();
        
        displayResults(data);
        
    } catch (error) {
        console.error("Error analyzing audio:", error, error.stack);
        statusDiv.textContent = `Error analyzing audio: ${error.message}. Please ensure the server is running and try again.`;
        statusDiv.style.backgroundColor = "#ffebee";
    } finally {
        loadingDiv.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});

// Display results
function displayResults(data) {
    if (!data || !data.features) {
        predictionResultDiv.textContent = "No results returned from analysis";
        statusDiv.style.backgroundColor = "#ffebee";
        return;
    }
    
    const prediction = data.prediction || 'No prediction available';
    const probability = data.probability != null ? (data.probability * 100).toFixed(2) : null;
    predictionResultDiv.textContent = `Analysis Result: ${prediction}${probability ? ` (Probability: ${probability}%)` : ''}`;
    
    featureTableBody.innerHTML = '';
    
    for (const [name, value] of Object.entries(data.features)) {
        const row = document.createElement('tr');
        
        const nameCell = document.createElement('td');
        nameCell.textContent = name;
        
        const valueCell = document.createElement('td');
        valueCell.textContent = typeof value === 'number' ? value.toFixed(6) : value;
        
        row.appendChild(nameCell);
        row.appendChild(valueCell);
        featureTableBody.appendChild(row);
    }
    
    resultsDiv.style.display = 'block';
    statusDiv.textContent = "Analysis complete. You can start a new recording or upload another file.";
    statusDiv.style.backgroundColor = "";
}

// Audio visualization
function startVisualization(stream) {
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    
    source.connect(analyser);
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function draw() {
        visualizationInterval = requestAnimationFrame(draw);
        
        analyser.getByteTimeDomainData(dataArray);
        
        canvasCtx.fillStyle = 'rgb(248, 248, 248)';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
        
        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = 'rgb(66, 133, 244)';
        canvasCtx.beginPath();
        
        const sliceWidth = canvas.width * 1.0 / bufferLength;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;
            
            if (i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }
            
            x += sliceWidth;
        }
        
        canvasCtx.lineTo(canvas.width, canvas.height / 2);
        canvasCtx.stroke();
    }
    
    draw();
}

function stopVisualization() {
    if (visualizationInterval) {
        cancelAnimationFrame(visualizationInterval);
        visualizationInterval = null;
    }
    
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
}

// Draw waveform
function drawWaveform() {
    if (!audioBuffer) return;
    
    const data = audioBuffer.getChannelData(0);
    const step = Math.ceil(data.length / canvas.width);
    const amp = canvas.height / 2;
    
    canvasCtx.fillStyle = 'rgb(248, 248, 248)';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    
    canvasCtx.lineWidth = 1;
    canvasCtx.strokeStyle = 'rgb(66, 133, 244)';
    canvasCtx.beginPath();
    
    for (let i = 0; i < canvas.width; i++) {
        let min = 1.0;
        let max = -1.0;
        
        for (let j = 0; j < step; j++) {
            const datum = data[(i * step) + j];
            if (datum < min) min = datum;
            if (datum > max) max = datum;
        }
        
        canvasCtx.moveTo(i, (1 + min) * amp);
        canvasCtx.lineTo(i, (1 + max) * amp);
    }
    
    canvasCtx.stroke();
}

// Encode to WAV
function encodeWAV(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitsPerSample = 16;
    
    const samples = audioBuffer.getChannelData(0);
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * bitsPerSample / 8, true);
    view.setUint16(32, numChannels * bitsPerSample / 8, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    
    return buffer;
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}
