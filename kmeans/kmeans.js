$(document).ready(function(){
    $("#traintest").on("change", function(){
        if($("#traintest").prop("checked"))
        {
            $(".plot").attr("src", "../kmeans/MallCustomers/test.svg?" + new Date().getTime());
        }
        else
        {
            $(".plot").attr("src", "../kmeans/MallCustomers/train.svg?" + new Date().getTime());
        }
    });
});
$(document).ready(function(){
    var n = 5;
    var i = "k-means++"
    var n_i = 20;
    $("#n_clusters").on("change", function(){
        n = $("#n_clusters").val();
    });
    $("input[name='init']").on("change", function(){
        i = $("input[name='init']:checked").val();
    });
    $("#n_init").on("change", function(){
        n_i = $("#n_init").val();
    });

  $(".run_button").click(function(){
    // alert(c);
    // alert(k);
    // alert(i);
    $.ajax({
            type: 'POST',
            url: "/cgi-bin/main.py",
            data: { algo:"kmeans", n_clusters:n, init:i, n_init:n_i }, //passing some input here
            dataType: "text",
            success: function(response){
               output = response;
               // alert(output);
            }
    	}).done(function(data){
        	console.log(data);
        	alert(data);
    	});
    $(".plot").attr("src", "../kmeans/MallCustomers/test.svg?" + new Date().getTime());
    $.get("../kmeans/MallCustomers/wcss.txt", function(data) {
     $("#wcss").text(data);});
    $.get("../kmeans/MallCustomers/time.txt", function(data) {
     $("#etime").text(data);});



 });
});
