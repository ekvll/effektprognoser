// const rect = document.getElementById("my-rect");
// rect.addEventListener("click", () => {
//   rect.setAttribute("fill", "black");
// });

const map = L.map("map").setView([59.3293, 18.0686], 13); // Stockholm

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

L.control
  .zoom({
    position: "topright",
  })
  .addTo(map);

// Legend
const legend = L.control({ position: "bottomright" });

legend.onAdd = function (map) {
  const div = L.DomUtil.create("div", "legend");
  div.innerHTML = "Legend";
  return div;
};

legend.addTo(map);

const legendDiv = document.querySelector(".legend");
legendDiv.innerHTML += "<br>item";

let boundaries = [
  { min: 10e6 - 1, max: Infinity },
  { min: 7, max: 10e6 - 1 },
  { min: 1, max: 7 },
  { min: 0.1, max: 1 },
  { min: -Infinity, max: 0.1 },
];

// For loop to iterate over number 0 to 4
for (let i = 0; i < 5; i++) {
  legendDiv.innerHTML += "<br>" + i;
}

// Loop over the min and max values in boundaries
for (let i = 0; i < boundaries.length; i++) {
  legendDiv.innerHTML += "<br>" + boundaries[i].min + " - " + boundaries[i].max;
}

const svgCode = `
<svg width="20" height="20" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <pattern id="diagonalLines" patternUnits="userSpaceOnUse" width="5" height="5" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="10" stroke="black" stroke-width="1" />
    </pattern>
  </defs>
  <rect width="20" height="20" fill="url(#diagonalLines)" />
</svg>
`;

legendDiv.innerHTML += "<br>" + svgCode;
