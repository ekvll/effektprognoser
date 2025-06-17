export function createDotPattern(color, id) {
  // This function creates a dot pattern for the map based on the provided color and ID
  // It creates a pattern element in the SVG namespace and appends it to the document
  // It first checks if a pattern with the same ID already exists and removes it if so
  // The pattern consists of a rectangle filled with the specified color and a circle
  // The rectangle is 8x8 pixels and the circle is drawn at the center with a radius of 0.75 pixels
  // The pattern is added to the SVG definitions section of the document
  // The function is used to create a pattern that can be used for styling features in the GeoJSON layer
  const svgNS = "http://www.w3.org/2000/svg";

  const oldPattern = document.getElementById(id);
  if (oldPattern) {
    oldPattern.remove();
  }

  const pattern = document.createElementNS(svgNS, "pattern");
  pattern.setAttribute("id", id);
  pattern.setAttribute("width", "8");
  pattern.setAttribute("height", "8");
  pattern.setAttribute("patternUnits", "userSpaceOnUse");

  const rect = document.createElementNS(svgNS, "rect");
  rect.setAttribute("width", "8");
  rect.setAttribute("height", "8");
  rect.setAttribute("fill", color || "transparent");

  const circle = document.createElementNS(svgNS, "circle");
  circle.setAttribute("cx", "3");
  circle.setAttribute("cy", "3");
  circle.setAttribute("r", "1.05");
  circle.setAttribute("fill", "black");

  pattern.appendChild(rect);
  pattern.appendChild(circle);

  let defs = document.querySelector("defs");
  if (!defs) {
    defs = document.createElementNS(svgNS, "defs");
    document.querySelector("svg").appendChild(defs);
  }
  defs.appendChild(pattern);
}

export function createPattern(color, id) {
  // This function creates a pattern for the map based on the provided color and ID
  // It creates a pattern element in the SVG namespace and appends it to the document
  // It first checks if a pattern with the same ID already exists and removes it if so
  // The pattern consists of a rectangle filled with the specified color and a diagonal line
  // The rectangle is 10x10 pixels and the line is drawn from the top-left to the bottom-right corner
  // The pattern is added to the SVG definitions section of the document
  // The function is used to create a pattern that can be used for styling features in the GeoJSON layer
  const svgNS = "http://www.w3.org/2000/svg";

  // Remove existing pattern with the same ID
  const oldPattern = document.getElementById(id);
  if (oldPattern) {
    // console.log("Removing old pattern with ID:", id);
    oldPattern.remove();
  }
  const pattern = document.createElementNS(svgNS, "pattern");
  pattern.setAttribute("id", id);
  pattern.setAttribute("width", "10");
  pattern.setAttribute("height", "10");
  pattern.setAttribute("patternUnits", "userSpaceOnUse");

  const rect = document.createElementNS(svgNS, "rect");
  rect.setAttribute("width", "10");
  rect.setAttribute("height", "10");
  rect.setAttribute("fill", color);

  const path = document.createElementNS(svgNS, "path");
  path.setAttribute("d", "M0,0 L10,10");
  path.setAttribute("style", "stroke:black; stroke-width:0.75");

  pattern.appendChild(rect);
  pattern.appendChild(path);

  const defs = document.querySelector("defs");
  defs.appendChild(pattern);
}

// import { addSvgPatternToDefs } from "./patterns.js";
// const map = L.map("map").setView([51.505, -0.09], 13);
// L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
//   maxZoom: 19,
// }).addTo(map);
//
// const polygonCoords = [
//   [51.51, -0.12],
//   [51.51, -0.06],
//   [51.5, -0.06],
// ];
//
// const polygon = L.polygon(polygonCoords).addTo(map);
//
// addSvgPatternToDefs(polygon, map);
//
// polygon.setStyle({
//   fill: true,
//   fillColor: "url(#diagonalLines)",
//   color: "#000",
//   weight: 2,
//   fillOpacity: 1,
// });
// export function addSvgPatternToDefs(polygon, map) {
//   const svg = map.getRenderer(polygon)._container;
//   let svgDefs = svg.querySelector("defs");
//   if (!svgDefs) {
//     svgDefs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
//     svg.insertBefore(svgDefs, svg.firstChild);
//   }
//
//   const pattern = document.createElementNS(
//     "http://www.w3.org/2000/svg",
//     "pattern",
//   );
//   pattern.setAttribute("id", "diagonalLines");
//   pattern.setAttribute("patternUnits", "userSpaceOnUse");
//   pattern.setAttribute("width", "10");
//   pattern.setAttribute("height", "10");
//   pattern.setAttribute("patternTransform", "rotate(45)");
//
//   const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
//   line.setAttribute("x1", "0");
//   line.setAttribute("y1", "0");
//   line.setAttribute("x2", "0");
//   line.setAttribute("y2", "10");
//   line.setAttribute("stroke", "black");
//   line.setAttribute("stroke-width", "1");
//
//   pattern.appendChild(line);
//   svgDefs.appendChild(pattern);
// }
//
// // SVG pattern of diagonal lines
// // const legendDiv = document.querySelector(".legend");
// // legendDiv.innerHTML += "<br>" + svgDiagonalLines;
// const svgDiagonalLines = `
// <svg width="20" height="20" xmlns="http://www.w3.org/2000/svg">
//   <defs>
//     <pattern id="diagonalLines" patternUnits="userSpaceOnUse" width="5" height="5" patternTransform="rotate(45)">
//       <line x1="0" y1="0" x2="0" y2="10" stroke="black" stroke-width="1" />
//     </pattern>
//   </defs>
//   <rect width="20" height="20" fill="url(#diagonalLines)" />
// </svg>
// `;
//
// export { svgDiagonalLines };
