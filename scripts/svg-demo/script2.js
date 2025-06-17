import { addSvgPatternToDefs } from "./patterns.js";

const map = L.map("map").setView([51.505, -0.09], 13);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
}).addTo(map);

const polygonCoords = [
  [51.51, -0.12],
  [51.51, -0.06],
  [51.5, -0.06],
];

const polygon = L.polygon(polygonCoords).addTo(map);

addSvgPatternToDefs(polygon, map);

polygon.setStyle({
  fill: true,
  fillColor: "url(#diagonalLines)",
  color: "#000",
  weight: 2,
  fillOpacity: 1,
});
