// Function to bring a layer to the front with a timer
export function bringLayerToFrontWithTimer(
  layer,
  intervalDuration,
  totalDuration,
) {
  // Calculate the total number of intervals
  let intervals = totalDuration / intervalDuration;

  // Initialize the interval
  const timer = setInterval(() => {
    // Bring the specified layer to the back
    layer.bringToFront();

    // Decrement the interval count
    if (--intervals <= 0) {
      clearInterval(timer); // Clear the interval after the total duration
    }
  }, intervalDuration);
}
