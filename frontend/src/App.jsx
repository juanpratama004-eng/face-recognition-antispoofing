import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import './App.css';

function App() {
  const webcamRef = useRef(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [faceBox, setFaceBox] = useState(null);
  const [result, setResult] = useState({ status: "idle", name: "-", is_real: false });

  const capture = async () => {
    if (isCameraOpen && webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        try {
          const response = await axios.post('http://localhost:8000/verify', { image: imageSrc });
          const data = response.data;
          setResult(data);
          if (data.status !== "no_face") {
            setFaceBox(data.box);
          } else {
            setFaceBox(null);
          }
        } catch (error) {
          console.error("API Error:", error);
        }
      }
    }
  };

  useEffect(() => {
    let interval;
    if (isCameraOpen) {
      interval = setInterval(capture, 600);
    } else {
      setFaceBox(null);
      setResult({ status: "idle", name: "-", is_real: false });
    }
    return () => clearInterval(interval);
  }, [isCameraOpen]);

  const getStatusInfo = () => {
    if (result.status === "no_face") return { class: "idle", text: "Mencari Wajah..." };
    if (result.status === "too_dark") return { class: "warning", text: "Terlalu Gelap" };
    if (result.status === "uncertain") return { class: "warning", text: "Kurang Jelas" };
    if (result.status === "error") return { class: "danger", text: "Kesalahan Sistem" };
    return result.is_real
      ? { class: "success", text: "Wajah Asli Terdeteksi" }
      : { class: "danger", text: "Tidak Terdeteksi" };
  };

  const getBoxColor = () => {
    if (result.status === 'spoof') return '#ef4444'; // Red-500
    if (result.name !== 'Unknown' && result.name !== 'Stranger' && result.name !== '-') return '#22c55e'; // Green-500
    return '#f59e0b'; // Amber-500
  };

  const statusInfo = getStatusInfo();

  return (
    <div className="app-container">
      <div className="header">
        <h1 className="app-title">Sistem Verifikasi Biometrik Wajah</h1>
        <h2 className="app-subtitle">Deteksi Liveness & Anti-Spoofing</h2>
      </div>

      {!isCameraOpen ? (
        <button className="primary-btn" onClick={() => setIsCameraOpen(true)}>
          Buka Kamera Verifikasi
        </button>
      ) : (
        <div className="main-content">
          <div className="controls">
            <button className="secondary-btn" onClick={() => setIsCameraOpen(false)}>
              Tutup Kamera
            </button>
          </div>

          <div className="camera-wrapper">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              className="webcam"
            />

            {faceBox && faceBox.length === 4 && (
              <div
                className="bounding-box"
                style={{
                  left: faceBox[0],
                  top: faceBox[1],
                  width: faceBox[2],
                  height: faceBox[3],
                  boxShadow: `inset 0 0 0 3px ${getBoxColor()}`,
                }}
              >
                <div
                  className="bounding-box-label"
                  style={{
                    backgroundColor: getBoxColor(),
                    color: (result.status === 'spoof' || result.name === 'Unknown' || result.name === 'Stranger') ? '#000' : '#fff'
                  }}
                >
                  {result.status === 'spoof' ? 'Spoofing Detected' : (result.name !== '-' ? result.name : 'Mencari..')}
                </div>
              </div>
            )}
          </div>

          <div className="status-panel">
            <div className={`status-indicator ${statusInfo.class}`}>
              {statusInfo.text}
            </div>
            <div className="identity">
              <span className="identity-label">Identitas Terdeteksi</span>
              <span className="identity-name">{result.name}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;