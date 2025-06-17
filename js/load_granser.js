import { map, selectedGranser } from "./map.js";

let currentNatagareGranser;
let currentKommunGranser;

export function loadNatagareGranser() {
  // This function loads the natagare granser layer onto the Leaflet map based on the selected values
  // It checks if the layer has already been created, and if not, it creates a new layer using the GeoJSON data
  // It styles the layer with a specific color, weight, and opacity
  const dataPath = `../data/geojson/background/Natomraden_ll84_dissolved.geojson`;
  // If the layer hasn't been created yet, load it first
  if (!currentNatagareGranser) {
    currentNatagareGranser = new L.GeoJSON.AJAX(dataPath, {
      style: function () {
        return {
          color: "#000",
          weight: 2,
          opacity: 0.7,
          fillOpacity: 0,
          fillColor: "#000",
          dashArray: "5, 5",
          dashOffset: "0",
        };
      },
    });

    // Only add layer to map if it's supposed to be visible
    if (selectedGranser.natagare) {
      currentNatagareGranser.addTo(map);
      console.log("Natagare granser layer added");
    }
  } else {
    // Toggle visibility
    if (selectedGranser.natagare) {
      if (!map.hasLayer(currentNatagareGranser)) {
        currentNatagareGranser.addTo(map);
        console.log("Natagare granser layer shown");
      }
    } else {
      if (map.hasLayer(currentNatagareGranser)) {
        map.removeLayer(currentNatagareGranser);
        console.log("Natagare granser layer hidden");
      }
    }
  }
}

export function loadKommunGranser() {
  // This function loads the kommun granser layer onto the Leaflet map based on the selected values
  // It checks if the layer has already been created, and if not, it creates a new layer using the GeoJSON data
  // It styles the layer with a specific color, weight, and opacity
  const dataPath = `../data/geojson/background/kommungranser_rss_skane_dissolved.geojson`;
  // If the layer hasn't been created yet, load it first
  if (!currentKommunGranser) {
    currentKommunGranser = new L.GeoJSON.AJAX(dataPath, {
      style: function () {
        return {
          color: "#000",
          weight: 2,
          opacity: 0.7,
          fillOpacity: 0,
          fillColor: "#000",
          dashArray: "0",
          dashOffset: "0",
        };
      },
    });

    // Only add layer to map if it's supposed to be visible
    if (selectedGranser.kommun) {
      currentKommunGranser.addTo(map);
      console.log("Kommun granser layer added");
    }
  } else {
    // Toggle visibility
    if (selectedGranser.kommun) {
      if (!map.hasLayer(currentKommunGranser)) {
        currentKommunGranser.addTo(map);
        console.log("Kommun granser layer shown");
      }
    } else {
      if (map.hasLayer(currentKommunGranser)) {
        map.removeLayer(currentKommunGranser);
        console.log("Kommun granser layer hidden");
      }
    }
  }
}

export { currentKommunGranser, currentNatagareGranser };
