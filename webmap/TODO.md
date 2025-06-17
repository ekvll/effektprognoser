## TO-DO

- [x] Make it so that, if `selectedValues.prognos` is either `ebd` or `elanvandning` and the square is labeled 'Ny bebyggelse' the value for that sqaure is shown below the diagonal lines.
- [ ] Reduce the number of functions needed in `styles.js`. Really only need one function for _Difference_ and one for _Percentage_.

  ```js
  function Difference(value, baseValue) {
      // Difference also takes baseValue as input.
      // baseValue is used as background color in
      // squares marked as "Ny laddinfra"/"Ny bebyggelse"
      ...
  }

  function Percentage(value) {
      ...
  }
  ```
