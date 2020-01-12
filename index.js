function includeHTML() {
  var z, i, elmnt, file, xhttp;
  /* Loop through a collection of all HTML elements: */
  z = document.getElementsByTagName("*");
  for (i = 0; i < z.length; i++) {
    elmnt = z[i];
    /*search for elements with a certain atrribute:*/
    file = elmnt.getAttribute("data-include-html");
    if (file) {
      /* Make an HTTP request using the attribute value as the file name: */
      xhttp = new XMLHttpRequest();
      xhttp.onreadystatechange = function() {
        if (this.readyState == 4) {
          if (this.status == 200) {elmnt.innerHTML = this.responseText;}
          if (this.status == 404) {elmnt.innerHTML = "Page not found.";}
          /* Remove the attribute, and call this function once more: */
          elmnt.removeAttribute("data-include-html");
          includeHTML();
        }
      }
      xhttp.open("GET", file, true);
      xhttp.send();
      /* Exit the function: */
      return;
    }
  }
}
$(document).ready(function(){
  $("#salary_data").click(function(){
    $(".plot").attr("src", "./LinearRegression/SalaryData/test.svg");
  });
});
$(document).ready(function(){
    $("#traintest").on("change", function(){
        if($("#traintest").prop("checked"))
        {
            $(".plot").attr("src", "./LinearRegression/SalaryData/test.svg");
        }
        else
        {
            $(".plot").attr("src", "./LinearRegression/SalaryData/train.svg");
        }
    });
});
// $(document).ready(function(){
//     $('.show_data').magnificPopup({
//       type: 'image',
//       overflowY: 'scroll',
//       fixedContentPos: 'true'
//     });
// });
// $(document).ready(includeHTML());
