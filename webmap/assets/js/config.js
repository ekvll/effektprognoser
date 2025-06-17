// Main configurations
// Change accordingly
export const mapConfig = {
  mapOptions: {
    center: [56.7591, 13.855], // where the map is centered (its mid-point)
    zoom: 8, // initial zoom level
    zoomControl: true,
    minZoom: 8, // how far out can the user zoom?
    maxZoom: 16, // how far in can the user zoom?
    attributionControl: false, // display Leaflet and Open Street Map in the lower right corner
  },
  maxBounds: [
    [53, 9],
    [61, 20],
  ], // the user is unable to pan outside this area
};

export const zoomThresholds = { 1: 8, 2: 10, 3: 12 }; // zoom thresholds used in layer control
