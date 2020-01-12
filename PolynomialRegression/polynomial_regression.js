$(document).ready(function(){
    $("#traintest").on("change", function(){
        if($("#traintest").prop("checked"))
        {
            $(".plot").attr("src", "./PositionSalaries/train.svg");
        }
        else
        {
            $(".plot").attr("src", "./PositionSalaries/train.svg");
        }
    });
});
$(document).ready(function(){
  $("#position_salaries").click(function(){
    $(".plot").attr("src", "./PositionSalaries/train.svg");
    $(".param").show()
    $("#mae").show()
  });
});
$(document).ready(function(){
     $("input[name='degree']").on("change", function(){
         var radioValue = $("input[name='degree']:checked").val();
         if(radioValue){
             $(".plot").attr("src", "./PositionSalaries/train_"+radioValue+".svg");
         }
         if (radioValue==='1') $('#mae').text("26695878787.878784");
         else if (radioValue==='2') $('#mae').text("6758833333.333336");
         else if (radioValue==='3') $('#mae').text("1515662004.662004");
         else if (radioValue==='4') $('#mae').text("210343822.8438155");
         else if (radioValue==='5') $('#mae').text("16382284.38228566");
     });

 });
