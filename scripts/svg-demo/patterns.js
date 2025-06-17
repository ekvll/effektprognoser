export function addSvgPatternToDefs(polygon, map) {
  const svg = map.getRenderer(polygon)._container;
  let svgDefs = svg.querySelector("defs");
  if (!svgDefs) {
    svgDefs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    svg.insertBefore(svgDefs, svg.firstChild);
  }

  const pattern = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "pattern",
  );
  pattern.setAttribute("id", "diagonalLines");
  pattern.setAttribute("patternUnits", "userSpaceOnUse");
  pattern.setAttribute("width", "10");
  pattern.setAttribute("height", "10");
  pattern.setAttribute("patternTransform", "rotate(45)");

  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", "0");
  line.setAttribute("y1", "0");
  line.setAttribute("x2", "0");
  line.setAttribute("y2", "10");
  line.setAttribute("stroke", "black");
  line.setAttribute("stroke-width", "1");

  pattern.appendChild(line);
  svgDefs.appendChild(pattern);
}
