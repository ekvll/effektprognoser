import {
  adjustBackgroundLayersOnZoom,
  onZoomEnd,
  bringLayersToFront,
} from "./layers.js";
import { loadDataLayer } from "./loader.js";
import { loadNatagareGranser, loadKommunGranser } from "./load_granser.js";
import { mapConfig } from "./config.js";
import { updateLegend2 } from "./legend.js"; // updateLegend is deprecated

// This module initializes a Leaflet map and sets up event listeners for a toolbox with radio buttons.
let map;
let selectedValues = {
  prognos: null,
  year: null,
  region: null,
  raps: null,
};
let selectedGranser = {
  kommun: null,
  natagare: null,
};
let legendControl = null;

function initializeMap() {
  // Initialize the map with specific options
  // This function sets the initial view, zoom level, and tile layer
  // It also sets the maximum bounds for the map
  // The map is centered on a specific location with a defined zoom level
  // The zoom control is added to the top right corner of the map
  // The OpenStreetMap tile layer is added to the map with attribution
  // The map is set to have a minimum zoom level of 8 and a maximum zoom level of 16
  // The map is initialized with a center point and zoom level
  // The map is set to have a maximum bounds to restrict the viewable area

  map = L.map("map", mapConfig.mapOptions);

  map.setMaxBounds(mapConfig.maxBounds);

  L.control
    .zoom({
      position: "topright",
    })
    .addTo(map);

  // const legendControl = createLegend();
  // legendControl.addTo(map);
  // Add the RISE copyright
  const customAttribution = L.control.attribution({
    position: "bottomright",
  });
  customAttribution.setPrefix(
    "© Region Skåne och RISE, Research Institutes of Sweden AB",
  );
  customAttribution.addTo(map);
}

function updateSelectedDataLayer() {
  selectedValues.prognos = document.querySelector(
    'input[name="prognos"]:checked',
  )?.value;
  selectedValues.year = document.querySelector(
    'input[name="year"]:checked',
  )?.value;
  selectedValues.region = document.querySelector(
    'input[name="region"]:checked',
  )?.value;
  selectedValues.raps = document.querySelector(
    'input[name="raps"]:checked',
  )?.value;

  console.log("Updated selected values:", selectedValues);
  loadDataLayer();

  if (legendControl) {
    legendControl.remove();
  }

  legendControl = updateLegend2();
  legendControl.addTo(map);
}

function eventsToolBox() {
  // This function sets up event listeners for the radio buttons in the toolbox
  // It listens for changes on all radio buttons and updates the selected values in the `selectedValues` object
  // When a radio button is changed, it logs the updated values and calls the `loadDataLayer` function to refresh the data layer on the map
  // The `selectedValues` object holds the currently selected values for prognos, year, region, and raps
  // The event listeners are added to all radio buttons with the type "radio" in the document
  // The `loadDataLayer` function is called to update the map with the new selections
  // The selected values are logged to the console for debugging purposes

  // Listen for changes on all radio buttons
  document.querySelectorAll('input[type="radio"]').forEach((item) => {
    item.addEventListener("change", () => {
      updateSelectedDataLayer();
      bringLayersToFront();
    });
  });

  // Listen for changes on the checkboxes for granser
  document.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
    cb.addEventListener("change", () => {
      selectedGranser.kommun = document.getElementById("kommungranser").checked;
      selectedGranser.natagare = document.getElementById("natagare").checked;

      loadKommunGranser();
      loadNatagareGranser();
    });
  });
}

// Initialize the map and events when the page loads
// This function is called when the DOM content is fully loaded
// It sets up the map and attaches event listeners to the toolbox
// The `DOMContentLoaded` event ensures that the map is initialized only after the HTML is fully parsed
// The `initializeMap` function is called to set up the map
// The `eventsToolBox` function is called to set up the event listeners for the toolbox
document.addEventListener("DOMContentLoaded", () => {
  initializeMap();
  eventsToolBox();
  updateSelectedDataLayer();
  adjustBackgroundLayersOnZoom();
});

export { map, selectedValues, selectedGranser };
