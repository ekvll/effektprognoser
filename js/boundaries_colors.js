import { selectedValues } from "./map.js";

export function getBoundariesAndColors() {
  // This function returns the boundaries and colors based on the selected prognos and raps values
  // It checks the selected prognos and raps values to determine which boundaries and colors to return
  // If the prognos is "effektbehov", it returns specific boundaries and colors based on the selected raps value
  //
  //  This function is also responsible for what colors and boundaries that are shown in the legend

  if (selectedValues.prognos == "effektbehov") {
    if (selectedValues.raps == "total") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 10, max: 10e6 - 1 },
          { min: 5, max: 10 },
          { min: 0.5, max: 5 },
          { min: -Infinity, max: 0.5 },
        ],
        colors: ["url(#hatch1)", "#060807", "#163019", "#296633", "#599E66"],
      };
    } else if (selectedValues.raps == "bostader") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 10, max: 10e6 - 1 },
          { min: 5, max: 10 },
          { min: 0.5, max: 5 },
          { min: -Infinity, max: 0.5 },
        ],
        colors: ["url(#hatch1)", "#3E2C13", "#C46737", "#EFA327", "#FEE292"],
      };
    } else if (selectedValues.raps == "industri_och_bygg") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 10, max: 10e6 - 1 },
          { min: 5, max: 10 },
          { min: 0.5, max: 5 },
          { min: -Infinity, max: 0.5 },
        ],
        colors: ["url(#hatch1)", "#4B1216", "#961B1E", "#ED1C29", "#F0675E"],
      };
    } else if (selectedValues.raps == "offentlig_och_privat_sektor") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 10, max: 10e6 - 1 },
          { min: 5, max: 10 },
          { min: 0.5, max: 5 },
          { min: -Infinity, max: 0.5 },
        ],
        colors: ["url(#hatch1)", "#10101D", "#393777", "#7E7DBC", "#BCBADD"],
      };
    } else if (selectedValues.raps == "transport") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 10, max: 10e6 - 1 },
          { min: 5, max: 10 },
          { min: 0.5, max: 5 },
          { min: -Infinity, max: 0.5 },
        ],
        colors: ["url(#hatch1)", "#091C1D", "#155384", "#068AB6", "#7BACAE"],
      };
    }

    // Effektbehov Difference
  } else if (selectedValues.prognos == "ebd") {
    if (selectedValues.raps == "total") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 2.5, max: 10e6 - 1 },
          { min: 0.8, max: 2.5 },
          { min: 0.2, max: 0.8 },
          { min: 0, max: 0.2 },
          { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#060807",
          "#163019",
          "#296633",
          "#599E66",
          "transparent",
        ],
      };
    } else if (selectedValues.raps == "bostader") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 0, max: 10e6 - 1 },
          { min: -0.01, max: 0 },
          { min: -0.1, max: -0.01 },
          { min: -10e10, max: -0.1 },
        ],
        // colors: ["url(#hatch1)", "#FEE292", "#EFA327", "#C46737", "#3E2C13"],
        colors: ["url(#hatch1)", "#3E2C13", "#C46737", "#EFA327", "#FEE292"],
        // boundaries: [
        //   { min: 10e6 - 1, max: Infinity },
        //   { min: 2.5, max: 10e6 - 1 },
        //   { min: 0.8, max: 2.5 },
        //   { min: 0.2, max: 0.8 },
        //   { min: 0, max: 0.2 },
        //   { min: -Infinity, max: 0 },
        // ],
        // colors: [
        //   "url(#hatch1)",
        //   "#3E2C13",
        //   "#C46737",
        //   "#EFA327",
        //   "#FEE292",
        //   "transparent",
        // ],
      };
    } else if (selectedValues.raps == "industri_och_bygg") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 2.5, max: 10e6 - 1 },
          { min: 0.8, max: 2.5 },
          { min: 0.2, max: 0.8 },
          { min: 0, max: 0.2 },
          { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#4B1216",
          "#961B1E",
          "#ED1C29",
          "#F0675E",
          "transparent",
        ],
      };
    } else if (selectedValues.raps == "offentlig_och_privat_sektor") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 2.5, max: 10e6 - 1 },
          { min: 0.8, max: 2.5 },
          { min: 0.2, max: 0.8 },
          { min: 0, max: 0.2 },
          { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#10101D",
          "#393777",
          "#7E7DBC",
          "#BCBADD",
          "transparent",
        ],
      };
    } else if (selectedValues.raps == "transport") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 2.5, max: 10e6 - 1 },
          { min: 0.8, max: 2.5 },
          { min: 0.2, max: 0.8 },
          { min: 0, max: 0.2 },
          // { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#091C1D",
          "#155384",
          "#068AB6",
          "#7BACAE",
          // "transparent",
        ],
      };
    }

    // Effektbehov Percentage
  } else if (selectedValues.prognos == "ebp") {
    if (selectedValues.raps == "total") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 350, max: 10e6 - 1 },
          { min: 50, max: 350 },
          { min: 20, max: 50 },
          { min: 0, max: 20 },
          { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#060807",
          "#163019",
          "#296633",
          "#599E66",
          "transparent",
        ],
      };
    } else if (selectedValues.raps == "bostader") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 0, max: 10e6 - 1 },
          { min: -2, max: 0 },
          { min: -4, max: -2 },
          { min: -10e10, max: -4 },
        ],
        // colors: ["url(#hatch1)", "#FEE292", "#EFA327", "#C46737", "#3E2C13"],
        colors: ["url(#hatch1)", "#3E2C13", "#C46737", "#EFA327", "#FEE292"],
        // boundaries: [
        //   { min: 10e6 - 1, max: Infinity },
        //   { min: 350, max: 10e6 - 1 },
        //   { min: 50, max: 350 },
        //   { min: 20, max: 50 },
        //   { min: 0, max: 20 },
        //   { min: -Infinity, max: 0 },
        // ],
        // colors: [
        //   "url(#hatch1)",
        //   "#3E2C13",
        //   "#C46737",
        //   "#EFA327",
        //   "#FEE292",
        //   "transparent",
        // ],
      };
    } else if (selectedValues.raps == "industri_och_bygg") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 50, max: 10e6 - 1 },
          { min: 20, max: 50 },
          { min: 10, max: 20 },
          { min: 0, max: 10 },
          { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#4B1216",
          "#961B1E",
          "#ED1C29",
          "#F0675E",
          "transparent",
        ],
      };
    } else if (selectedValues.raps == "offentlig_och_privat_sektor") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 25, max: 10e6 - 1 },
          { min: 15, max: 25 },
          { min: 5, max: 15 },
          { min: 0, max: 5 },
          { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#10101D",
          "#393777",
          "#7E7DBC",
          "#BCBADD",
          "transparent",
        ],
      };
    } else if (selectedValues.raps == "transport") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 1600, max: 10e6 - 1 },
          { min: 800, max: 1600 },
          { min: 200, max: 800 },
          { min: 0, max: 200 },
          // { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#091C1D",
          "#155384",
          "#068AB6",
          "#7BACAE",
          // "transparent",
        ],
      };
    }

    // Elanvandning
  } else if (selectedValues.prognos == "ead") {
    if (selectedValues.raps == "total") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 10000, max: 10e6 - 1 },
          { min: 1000, max: 10000 },
          { min: 50, max: 1000 },
          { min: 0, max: 50 },
          { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#060807",
          "#163019",
          "#296633",
          "#599E66",
          "transparent",
        ],
      };
    } else if (selectedValues.raps == "bostader") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 0, max: 10e6 - 1 },
          { min: -25, max: 0 },
          { min: -50, max: -25 },
          { min: -10e10, max: -50 },
          //{ min: -Infinity, max: -2000 },
        ],
        // colors: [
        //   "url(#hatch1)",
        //   "#3E2C13",
        //   "#C46737",
        //   "#EFA327",
        //   "#FEE292",
        //   "transparent",
        // ],
        // colors: ["url(#hatch1)", "#FEE292", "#EFA327", "#C46737", "#3E2C13"],
        colors: ["url(#hatch1)", "#3E2C13", "#C46737", "#EFA327", "#FEE292"],
      };
    } else if (selectedValues.raps == "industri_och_bygg") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 50000, max: 10e6 - 1 },
          { min: 5000, max: 50000 },
          { min: 20, max: 5000 },
          { min: 0, max: 20 },
          { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#4B1216",
          "#961B1E",
          "#ED1C29",
          "#F0675E",
          "transparent",
        ],
      };
    } else if (selectedValues.raps == "offentlig_och_privat_sektor") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 10000, max: 10e6 - 1 },
          { min: 1000, max: 10000 },
          { min: 50, max: 1000 },
          { min: 0, max: 50 },
          { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#10101D",
          "#393777",
          "#7E7DBC",
          "#BCBADD",
          "transparent",
        ],
      };
    } else if (selectedValues.raps == "transport") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 10000, max: 10e6 - 1 },
          { min: 1000, max: 10000 },
          { min: 50, max: 1000 },
          { min: 0, max: 50 },
          // { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#091C1D",
          "#155384",
          "#068AB6",
          "#7BACAE",
          // "transparent",
        ],
      };
    }

    // Elanvandning Percentage
  } else if (selectedValues.prognos == "eap") {
    if (selectedValues.raps == "total") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 350, max: 10e6 - 1 },
          { min: 50, max: 350 },
          { min: 20, max: 50 },
          { min: 0, max: 20 },
          { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#060807",
          "#163019",
          "#296633",
          "#599E66",
          "transparent",
        ],
      };
    } else if (selectedValues.raps == "bostader") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 0, max: 10e6 - 1 },
          { min: -2, max: 0 },
          { min: -6, max: -2 },
          { min: -10e10, max: -6 },
          // { min: -Infinity, max: -30 },
        ],
        // colors: ["url(#hatch1)", "#FEE292", "#EFA327", "#C46737", "#3E2C13"],
        colors: ["url(#hatch1)", "#3E2C13", "#C46737", "#EFA327", "#FEE292"],
      };
    } else if (selectedValues.raps == "industri_och_bygg") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 350, max: 10e6 - 1 },
          { min: 50, max: 350 },
          { min: 20, max: 50 },
          { min: 0, max: 20 },
          { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#4B1216",
          "#961B1E",
          "#ED1C29",
          "#F0675E",
          "transparent",
        ],
      };
    } else if (selectedValues.raps == "offentlig_och_privat_sektor") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 350, max: 10e6 - 1 },
          { min: 50, max: 350 },
          { min: 20, max: 50 },
          { min: 0, max: 20 },
          { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#10101D",
          "#393777",
          "#7E7DBC",
          "#BCBADD",
          "transparent",
        ],
      };
    } else if (selectedValues.raps == "transport") {
      return {
        boundaries: [
          { min: 10e6 - 1, max: Infinity },
          { min: 1600, max: 10e6 - 1 },
          { min: 800, max: 1600 },
          { min: 200, max: 800 },
          { min: 0, max: 200 },
          // { min: -Infinity, max: 0 },
        ],
        colors: [
          "url(#hatch1)",
          "#091C1D",
          "#155384",
          "#068AB6",
          "#7BACAE",
          // "transparent",
        ],
      };
    }
  }
}
