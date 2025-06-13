import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import sharp from 'sharp';
import supabase from '../lib/supabase.js';
import { v4 as uuidv4 } from 'uuid';

// Untuk ES module (__dirname)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Helper function untuk handle Python process
// Helper function SIMPLE VERSION
const handlePythonProcess = (python, res, tempFilePath = null) => {
  let output = '';
  let finished = false;

  const finish = (data, code = 200) => {
    if (finished) return;
    finished = true;
    
    // Cleanup
    if (tempFilePath && fs.existsSync(tempFilePath)) {
      fs.unlinkSync(tempFilePath);
    }
    
    // Kill process
    try { python.kill(); } catch (e) {}
    
    // Send response
    try {
      res.status(code).json(data);
    } catch (e) {
      console.error('Response error:', e.message);
    }
  };

  python.stdout.on('data', (data) => output += data.toString());
  python.stderr.on('data', () => {}); // Ignore all stderr
  
  python.on('close', (code) => {
    if (code === 0 && output.trim()) {
      try {
        finish(JSON.parse(output.trim()));
      } catch (e) {
        finish({error: 'Parse error'}, 500);
      }
    } else {
      finish({error: 'Process failed'}, 500);
    }
  });

  setTimeout(() => finish({error: 'Timeout'}, 500), 30000);
};

// Tabular prediction
export const predictTabular = (req, res) => {
  const features = req.body.features;

  if (!features || !Array.isArray(features)) {
    return res.status(400).json({
      error: 'Features array is required',
      status: 'error',
    });
  }

  const scriptPath = path.join(__dirname, '..', 'models', 'predict.py');
  const python = spawn('python', [scriptPath, JSON.stringify(features)], {
    stdio: ['ignore', 'pipe', 'pipe'] // ignore stdin untuk tabular
  });

  handlePythonProcess(python, res);
};

// Image prediction - FIXED VERSION
export const predictImage = (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      error: 'Image file is required',
      status: 'error',
    });
  }

  // Create temp directory if not exists
  const tempDir = path.join(__dirname, '..', 'temp');
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }

  // Generate unique temp file path
  const tempFileName = `temp_image_${uuidv4()}.jpg`;
  const tempFilePath = path.join(tempDir, tempFileName);

  try {
    // Save uploaded file buffer to temporary file
    fs.writeFileSync(tempFilePath, req.file.buffer);
    console.log('Saved temp image:', tempFilePath);

    const scriptPath = path.join(__dirname, '..', 'models', 'predict.py');
    
    // Spawn Python process with file path as argument
    const python = spawn('python', [scriptPath, 'image', tempFilePath], {
      stdio: ['ignore', 'pipe', 'pipe'], // IMPORTANT: ignore stdin completely
      shell: process.platform === 'win32', // Enable shell for Windows
      detached: false // Keep attached to parent process
    });

    // Handle process with cleanup
    handlePythonProcess(python, res, tempFilePath);

  } catch (error) {
    // Cleanup on error
    if (fs.existsSync(tempFilePath)) {
      try {
        fs.unlinkSync(tempFilePath);
      } catch (cleanupErr) {
        console.warn('Failed to cleanup temp file after error:', cleanupErr.message);
      }
    }

    res.status(500).json({
      error: 'Failed to process image: ' + error.message,
      status: 'error',
    });
  }
};