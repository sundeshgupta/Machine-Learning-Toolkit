$(document).ready(function(){
    $("#traintest").on("change", function(){
        if($("#traintest").prop("checked"))
        {
            $(".plot").attr("src", "../test.svg?" + new Date().getTime());
        }
        else
        {
            $(".plot").attr("src", "../train.svg?" + new Date().getTime());
        }
    });
  $("#salary_data").click(function(){

    $('#mae').text("23370078.800832972");
  });

  $(".rangeslider").on('change', (function(){
  // alert("Submitted");
  // $("#theForm").submit();
    var val = $(".rangeslider").val();
    // alert(val);
    $.ajax({
        type: 'POST',
        url: "/cgi-bin/simple_linear_regression.py",
        data: {param: val}, //passing some input here
        dataType: "text",
        success: function(response){
           output = response;
           alert(output);
        }
	}).done(function(data){
    	console.log(data);
    	alert(data);
        (".plot").attr("src", "../test.svg?" + new Date().getTime());
	})
    (".plot").attr("src", "../test.svg?" + new Date().getTime());


  // $(".plot").attr("src", "../test.svg");
  }));
});
