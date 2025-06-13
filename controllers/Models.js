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
const handlePythonProcess = (python, res, tempFilePath = null) => {
  let output = '';
  let responded = false;
  let processEnded = false;

  const cleanup = () => {
    if (tempFilePath && fs.existsSync(tempFilePath)) {
      try {
        fs.unlinkSync(tempFilePath);
        console.log('Cleaned up temp file:', tempFilePath);
      } catch (err) {
        console.warn('Failed to cleanup temp file:', err.message);
      }
    }
  };

  const safeRespond = (responseData, statusCode = 200) => {
    if (responded || res.headersSent) {
      console.warn('Attempted to send response after headers already sent');
      return false;
    }
    
    responded = true;
    cleanup();
    
    try {
      if (statusCode === 200) {
        res.json(responseData);
      } else {
        res.status(statusCode).json(responseData);
      }
      return true;
    } catch (err) {
      console.error('Error sending response:', err.message);
      return false;
    }
  };

  const safeKill = () => {
    if (!processEnded) {
      processEnded = true;
      try {
        python.kill('SIGTERM');
        // Fallback kill after 2s
        setTimeout(() => {
          if (python.exitCode === null) {
            python.kill('SIGKILL');
          }
        }, 2000);
      } catch (err) {
        console.warn('Error killing Python process:', err.message);
      }
    }
  };

  python.stdout.on('data', (data) => {
    if (!responded) {
      output += data.toString();
    }
  });

  python.stderr.on('data', (data) => {
    const errMsg = data.toString();
    
    // Skip TensorFlow info messages completely - don't even log them
    const isTensorFlowInfo = 
      errMsg.includes('INFO:') ||
      errMsg.includes('Created TensorFlow Lite') ||
      errMsg.includes('This TensorFlow binary is optimized') ||
      errMsg.includes('oneDNN') ||
      errMsg.includes('XNNPACK');

    if (!isTensorFlowInfo) {
      console.error('Python stderr:', errMsg);
    }

    // Only treat as real error if it contains actual error indicators
    const isRealError =
      errMsg.includes('Traceback') ||
      errMsg.includes('Error:') ||
      errMsg.includes('Exception:') ||
      errMsg.includes('ModuleNotFoundError') ||
      errMsg.includes('FileNotFoundError');

    if (!responded && isRealError && !isTensorFlowInfo) {
      safeKill();
      safeRespond({
        error: 'Model prediction failed: ' + errMsg,
        status: 'error',
      }, 500);
    }
  });

python.on('close', (code) => {
  processEnded = true;
  clearTimeout(timeoutId); // ← tambahkan ini di awal

  if (responded || res.headersSent) {
    return; // Already responded
  }

  try {
    if (code === 0 && output.trim()) {
      const result = JSON.parse(output.trim());
      return safeRespond(result);
    } else {
      return safeRespond({
        error: `Python process exited with code ${code}`,
        status: 'error',
        output: output
      }, 500);
    }
  } catch (err) {
    return safeRespond({
      error: 'Failed to parse output: ' + err.message,
      status: 'error',
      raw_output: output
    }, 500);
  }
});


  python.on('error', (err) => {
    processEnded = true;
    
    if (!responded && !res.headersSent) {
      safeRespond({
        error: 'Failed to start Python process: ' + err.message,
        status: 'error',
      }, 500);
    }
  });

  // Disable stdin completely to prevent EPIPE
if (python.stdin) {
  try {
    python.stdin.on('error', (err) => {
      console.warn('Python stdin error (ignored):', err.message);
    });
    python.stdin.end();
  } catch (err) {
    console.warn('Error handling stdin:', err.message);
  }
} else {
  console.log('stdin is null — skipped .on and .end');
}


  // Timeout handler
  const timeoutId = setTimeout(() => {
    if (!responded && !res.headersSent) {
      safeKill();
      safeRespond({
        error: 'Python script timeout (30s)',
        status: 'error',
      }, 500);
    }
  }, 30000);

  // Clear timeout if process ends normally
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