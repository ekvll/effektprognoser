import { map, selectedValues } from "./map.js";
import { zoomThresholds } from "./config.js";
import {
  loadTileLayer,
  loadRoads,
  loadLakes,
  loadCities,
  loadCitiesHiRes,
  loadDanmark,
  loadRoadsBoundary,
  loadRoadsHiRes,
  loadRoadsBoundaryHiRes,
  currentLayer,
  danmarkLayer,
  tileLayer,
  citiesLayer,
  citiesLayerHiRes,
  roadsLayerBoundary,
  roadsLayer,
  roadsLayerHiRes,
  roadsLayerBoundaryHiRes,
  lakesLayer,
} from "./loader.js";
import { bringLayerToFrontWithTimer } from "./utils.js";

export function bringLayersToFront() {
  const layers = [
    roadsLayerBoundary,
    roadsLayer,
    roadsLayerBoundaryHiRes,
    roadsLayerHiRes,
    lakesLayer,
  ];
  for (let i = 0; i < layers.length; i++) {
    let layer = layers[i];
    if (map.hasLayer(layer)) {
      bringLayerToFrontWithTimer(layer, 500, 2000);
    }
  }
}

function addRemoveLayer(zoomLevel, layer, addFunc, th, reverse) {
  // Add or remove a layer from the map depending on current zoom level,
  // zoom threshold and wether the function should be used in a reverse manner
  //
  // This docstring needs improvements!

  // Define thw upper and lower thresholds
  let highTh;

  if (th + 1 <= Object.keys(zoomThresholds).length) {
    highTh = th + 1;
  } else {
    highTh = th;
  }

  let lowTh;

  if (th - 1 >= 0) {
    lowTh = th - 1;
  } else {
    lowTh = th;
  }

  // If reverse
  if (reverse === 0) {
    if (highTh !== th) {
      if (
        zoomLevel < zoomThresholds[highTh] &&
        zoomLevel >= zoomThresholds[th]
      ) {
        if (!map.hasLayer(layer)) {
          addFunc();
        }
      } else {
        if (map.hasLayer(layer)) {
          map.removeLayer(layer);
        }
      }
    } else {
      if (zoomLevel >= zoomThresholds[th]) {
        if (!map.hasLayer(layer)) {
          addFunc();
        }
      } else {
        if (map.hasLayer(layer)) {
          map.removeLayer(layer);
        }
      }
    }

    // If not reverse
  } else {
    if (zoomLevel < zoomThresholds[th]) {
      if (!map.hasLayer(layer)) {
        addFunc();
      }
    } else {
      if (map.hasLayer(layer)) {
        map.removeLayer(layer);
      }
    }
  }
}

export function onZoomEnd() {
  // This function adjusts the visibility (add/remove) of layers based on the current zoom level
  // The higher the zoom level the more zoomed in you are
  // Zoom level equal to 12 is more zoomed in than zoom level equal to 10
  const zoomLevel = map.getZoom();

  addRemoveLayer(zoomLevel, tileLayer, loadTileLayer, 3, 0);
  addRemoveLayer(
    zoomLevel,
    roadsLayerBoundaryHiRes,
    loadRoadsBoundaryHiRes,
    2,
    0,
  );
  addRemoveLayer(zoomLevel, roadsLayerHiRes, loadRoadsHiRes, 2, 0);
  addRemoveLayer(zoomLevel, citiesLayer, loadCities, 2, 1);
  addRemoveLayer(zoomLevel, citiesLayerHiRes, loadCitiesHiRes, 2, 0);
  addRemoveLayer(zoomLevel, lakesLayer, loadLakes, 3, 1);
  addRemoveLayer(zoomLevel, danmarkLayer, loadDanmark, 3, 1);
  addRemoveLayer(zoomLevel, roadsLayerBoundary, loadRoadsBoundary, 2, 1);
  addRemoveLayer(zoomLevel, roadsLayer, loadRoads, 2, 1);
}

export function adjustBackgroundLayersOnZoom() {
  map.on("zoomend", onZoomEnd);

  onZoomEnd();
}
