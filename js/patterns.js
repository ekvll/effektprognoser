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
