$(document).ready(function(){
    $("#traintest").on("change", function(){
        if($("#traintest").prop("checked"))
        {
            $(".plot").attr("src", "./SalaryData/test.svg");
        }
        else
        {
            $(".plot").attr("src", "./SalaryData/train.svg");
        }
    });
});
$(document).ready(function(){
  $("#salary_data").click(function(){
    $(".plot").attr("src", "./plot10.svg");
    $.get("./plot10.txt", function(data) {
     $("#mse").text(data);
    }, 'text');
    $.get("./plottime10.txt", function(data) {
     $("#etime").text(data);
    }, 'text');
    $.get("./plotcoef10.txt", function(data) {
     $("#coef").text(data);
    }, 'text');
    $.get("./plotint10.txt", function(data) {
     $("#intercept").text(data);
    }, 'text');
  });
  $(".poly_button").click(function(){
     $(".adv").show();
     $(".gauss_button").css('color', 'white');
     $(".poly_button").css('color', 'red');
     $("input[name='degree']").on("change", function(){
         var radioValue = $("input[name='degree']:checked").val();
         if(radioValue){
             $(".plot").attr("src", "./trainp"+radioValue+"0.svg");
             $.get("./trainp"+radioValue+"0.txt", function(data) {
              $("#mse").text(data);
             }, 'text');
             $.get("./traintimep"+radioValue+"0.txt", function(data) {
              $("#etime").text(data);
             }, 'text');
             $.get("./trainintp"+radioValue+"0.txt", function(data) {
              $("#intercept").text(data);
             }, 'text');
             $.get("./traincoefp"+radioValue+"0.txt", function(data) {
              $("#coef").text(data);
             }, 'text');
             $(".show_basis").attr("href", "./trainbasisp"+radioValue+"0.svg")
         }
     });
     var x = "";
     $("input[name='reg']").on("change", function(){
        var checkedValue = $("input[name='reg']:checked").val();
        if (checkedValue)
        {
            $(".regularisation").show();
        }
        else
        {
            $(".regularisation").hide();
        }
      });

      $(".ridge").click(function(){
          x="r";
          $(".lasso").css('color', 'white');
          $(".ridge").css('color', 'red');
      });

      $(".lasso").click(function(){
          x="l";
          $(".lasso").css('color', 'red');
          $(".ridge").css('color', 'white');
      });
      $("input[name='alpha']").on("change", function(){
               var radioValue1 = $("input[name='alpha']:checked").val();
               var radioValue = $("input[name='degree']:checked").val();
               if( radioValue1 ){
                   // $(".plot").attr("src", "./trainp"+radioValue+x+radioValue1+".svg");
                   $.get("./trainp"+radioValue+x+radioValue1+".txt", function(data) {
                   $("#mse").text(data);
                    });
                    $.get("./traintimep"+radioValue+x+radioValue1+".txt", function(data) {
                    $("#etime").text(data);
                     });
                    $(".show_basis").attr("src", "./trainbasisp"+radioValue+x+radioValue1+".svg");
               }
           });
  });


  $(".gauss_button").click(function(){
     $(".adv").show();
     $(".gauss_button").css('color', 'red');
     $(".poly_button").css('color', 'white');
      $("input[name='degree']").on("change", function(){
         var radioValue = $("input[name='degree']:checked").val();
         if(radioValue){
             $(".plot").attr("src", "./traing"+radioValue+"0.svg");
             $.get("./traing"+radioValue+"0.txt", function(data) {
              $("#mse").text(data);
             $.get("./traintimeg"+radioValue+"0.txt", function(data) {
              $("#etime").text(data);
             }, 'text');

             $.get("./trainintg"+radioValue+"0.txt", function(data) {
              $("#intercept").text(data);
             }, 'text');
             $.get("./traincoefg"+radioValue+"0.txt", function(data) {
              $("#coef").text(data);
             }, 'text');
             $(".show_basis").attr("href", "./trainbasisg"+radioValue+"0.svg")
          });
         }
     });
     var x = "";
     $("input[name='reg']").on("change", function(){
        var checkedValue = $("input[name='reg']:checked").val();
        if (checkedValue)
        {
            $(".regularisation").show();
        }
        else
        {
            $(".regularisation").hide();
        }
      });

      $(".ridge").click(function(){
          x="r";
          $(".lasso").css('color', 'white');
          $(".ridge").css('color', 'red');
      });

      $(".lasso").click(function(){
          x="l";
          $(".lasso").css('color', 'red');
          $(".ridge").css('color', 'white');
      });
      $("input[name='alpha']").on("change", function(){
               var radioValue1 = $("input[name='alpha']:checked").val();
               var radioValue = $("input[name='degree']:checked").val();
               if( radioValue1 ){
                   $(".plot").attr("src", "./traing"+radioValue+x+radioValue1+".svg");
                   $.get("./traing"+radioValue+x+radioValue1+".txt", function(data) {
                   $("#mse").text(data);
                    });
                    $.get("./traintimeg"+radioValue+x+radioValue1+".txt", function(data) {
                    $("#etime").text(data);
                     });
                     $(".show_basis").attr("src", "./trainbasisg"+radioValue+x+radioValue1+".svg");
                }
           });
  });
});
