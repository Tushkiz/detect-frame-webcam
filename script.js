import * as tf from "@tensorflow/tfjs";
import * as poseDetection from "@tensorflow-models/pose-detection";

const model = poseDetection.SupportedModels.MoveNet;
const detector = await poseDetection.createDetector(model);
const $video = document.querySelector("video");
const $canvasContainer = document.querySelector(".canvas-wrapper");
const $canvas = document.querySelector("canvas");
const $status = document.querySelector(".status");

async function enableCamera() {
  // getUsermedia parameters to force video but not audio.
  const constraints = {
    video: true,
  };

  // Activate the webcam stream.
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  $video.srcObject = stream;

  await new Promise((resolve) => {
    $video.addEventListener("loadeddata", resolve);
  });
  $video.play();

  const videoWidth = $video.videoWidth;
  const videoHeight = $video.videoHeight;

  $video.width = videoWidth;
  $video.height = videoHeight;
  $canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;
  $canvas.width = videoWidth;
  $canvas.height = videoHeight;

  $status.style.visibility = "visible";
}

function renderResult(poses) {
  if (!poses) return;
  const keypoints = poses.keypoints;
  const ctx = $canvas.getContext("2d");

  const videoWidth = $canvas.width;
  const videoHeight = $canvas.height;

  ctx.drawImage($video, 0, 0, videoWidth, videoHeight);
  ctx.fillStyle = "Green";

  const leftEye = keypoints[1];
  const rightEye = keypoints[2];
  const leftShoulder = keypoints[5];
  const rightShoulder = keypoints[6];

  if (window.DRAW_KEYPOINTS) {
    [leftEye, rightEye, leftShoulder, rightShoulder].forEach((keypoint) => {
      if (keypoint.score < 0.5) return;
      const circle = new Path2D();
      circle.arc(keypoint.x, keypoint.y, 8, 0, 2 * Math.PI);
      ctx.fill(circle);
    });
  }

  const ready = [leftEye, rightEye, leftShoulder, rightShoulder].every(
    (keypoint) => {
      return keypoint.score > 0.5;
    }
  );
  if (ready) {
    $status.classList.add("ready");
    $status.textContent = "";
  } else {
    $status.classList.remove("ready");
    $status.textContent = "Ensure your head and shoulders are in shot";
  }
}

async function runModel() {
  try {
    const [poses] = await detector.estimatePoses($video, {
      flipHorizontal: true,
    });

    renderResult(poses);
    window.requestAnimationFrame(runModel);
  } catch (error) {
    console.log("Something went wrong!\n", error);
    detector.dispose();
  }
}

await enableCamera();
runModel();
