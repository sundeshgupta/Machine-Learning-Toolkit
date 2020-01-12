var dataset = "";
var directory = "";
$(document).ready(function(){
    $("#traintest").on("change", function(){
        if($("#traintest").prop("checked"))
        {
            $(".plot").attr("src", "../knn/"+directory+"/test.svg?" + new Date().getTime());
        }
        else
        {
            $(".plot").attr("src", "../knn/"+directory+"/train.svg?" + new Date().getTime());
        }
    });
});
$(document).ready(function(){
    var t = 20;
    var n = 5;
    $("#t_size").on("change", function(){
            t = $("#t_size").val();
        });
    $("#n_neighbours").on("change", function(){
            n = $("#n_neighbours").val();
        });
    $("#select_data").on("change", function(){
            dataset = $("#select_data").val();
            if (dataset==="Social_Network_Ads.csv") directory = "SocialNetworkAds";
            else if (dataset === "moons") directory = "Moons";
            else if (dataset === "circles") directory = "Circles";
                $(".show_data").attr("href", "../knn/" + directory + "/data.png");  
        });


  $(".run_button").click(function(){
    // alert(c);
    // alert(k);
    // alert(t);
    $.ajax({
            type: 'POST',
            url: "/cgi-bin/main.py",
            data: { algo:"knn", dataset:dataset, t_size:t, n_neighbours:n }, //passing some input here
            dataType: "text",
            success: function(response){
               output = response;
               // alert(output);
            }
    	}).done(function(data){
        	console.log(data);
        	alert(data);
    	});
    $(".plot").attr("src", "../knn/"+directory+"/test.svg?" + new Date().getTime());
    $.get("../knn/"+directory+"/intercept.txt", function(data) {
     $("#intercept").text(data);});
    $.get("../knn/"+directory+"/coef.txt", function(data) {
     $("#coef").text(data);});
    $.get("../knn/"+directory+"/accu.txt", function(data) {
     $("#accu").text(data);});
    $.get("../knn/"+directory+"/time.txt", function(data) {
     $("#etime").text(data);});



 });
});
