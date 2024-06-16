const express = require('express');
const multer = require('multer');
const fs = require('fs'); // Standard fs module for synchronous functions
const fsPromises = fs.promises; // Promises-based fs functions
const path = require('path');
const faceapi = require('face-api.js');
const canvas = require('canvas');
const { Canvas, Image, ImageData, loadImage } = canvas;
require('dotenv').config();
const bodyParser = require('body-parser');
const twilio = require('twilio');

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const PORT = 9000;

app.use(bodyParser.json());
app.use(express.static('public'));

const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_AUTH_TOKEN;
const client = twilio(accountSid, authToken);

app.post('/send-sms', (req, res) => {
  const { to, message } = req.body;

  client.messages
    .create({
      body: message,
      from: process.env.TWILIO_PHONE_NUMBER,
      to: to
    })
    .then(message => {
      console.log(`SMS sent: ${message.sid}`);
      res.send('SMS sent successfully');
    })
    .catch(error => {
      console.error('Error sending SMS:', error);
      res.status(500).send('Error sending SMS');
    });
});

// Load face-api.js models when server starts
async function loadFaceApiModels() {
    console.log('Loading face-api.js models...');
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'public', 'models'));
    await faceapi.nets.faceLandmark68Net.loadFromDisk(path.join(__dirname, 'public', 'models'));
    await faceapi.nets.faceRecognitionNet.loadFromDisk(path.join(__dirname, 'public', 'models'));
    console.log('Face-api.js models loaded.');
  }
  
  loadFaceApiModels();

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const label = req.query.label;
    const dir = path.join(__dirname, 'public', 'uploads', label);
    console.log(`Checking existence of directory: ${dir}`);
    if (!fs.existsSync(dir)) {
      console.log(`Directory does not exist. Creating: ${dir}`);
      fs.mkdirSync(dir, { recursive: true });
    }
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    const filename = `${Date.now()}.jpg`;
    console.log(`Saving file as: ${filename}`);
    cb(null, filename);
  }
});

const upload = multer({ storage });

app.post('/upload', upload.single('image'), (req, res) => {
  console.log('Upload route hit. Files uploaded:');
  console.log('file: ', req.file);
  res.send('Image uploaded');
});

let labeledFaceDescriptors = [];

async function loadLabeledImages() {
  try {
    const labels = await fsPromises.readdir(path.join(__dirname, 'public', 'uploads'));
    console.log('Labels found:', labels);

    for (const label of labels) {
      const descriptors = [];
      const images = await fsPromises.readdir(path.join(__dirname, 'public', 'uploads', label));
      console.log('Images found for label', label, ':', images);

      for (const image of images) {
        const imgPath = path.join(__dirname, 'public', 'uploads', label, image);
        console.log(`Loading image: ${imgPath}`);
        const img = await loadImageFromFile(imgPath);
        console.log("done with loadImageFromFile");

        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
        console.log("done with detections");
        if (detections) {
          descriptors.push(detections.descriptor);
        }
      }

      if (descriptors.length > 0) {
        labeledFaceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors));
      }
    }
    console.log('Processed labeled face descriptors:', labeledFaceDescriptors);
    return labeledFaceDescriptors;
  } catch (error) {
    console.error('Error loading labeled images:', error);
  }
}

async function loadImageFromFile(filePath) {
  try {
    console.log(`Reading image file: ${filePath}`);
    const img = await loadImage(filePath);
    console.log(`Image loaded: ${filePath}`);
    return img;
  } catch (error) {
    console.error(`Error reading image file: ${filePath}`, error);
    throw error;
  }
}

// Serve HTML files
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/register', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'register.html'));
});

app.get('/real-time-face-recognition', async (req, res) => {
  // Ensure labeled face descriptors are loaded before serving the page
  if (labeledFaceDescriptors.length === 0) {
    console.log('No labeled face descriptors found. Loading from images...');
    labeledFaceDescriptors = await loadLabeledImages();
    console.log('Labeled face descriptors loaded.');
  } else {
    console.log('Labeled face descriptors already loaded.');
  }
  res.sendFile(path.join(__dirname, 'public', 'real_time_face_recognition.html'));
});

app.get('/get-labeled-faces', (req, res) => {
  console.log('Sending labeled face descriptors to client.');
  res.json(labeledFaceDescriptors.map(desc => ({
    label: desc.label,
    descriptors: desc.descriptors.map(d => Array.from(d))
  })));
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
