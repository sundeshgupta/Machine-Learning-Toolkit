
var dataset = "";
var directory = "";
$(document).ready(function(){
    $("#traintest").on("change", function(){
        if($("#traintest").prop("checked"))
        {
            $(".plot").attr("src", "../LogisticRegression/"+directory+"/test.svg?" + new Date().getTime());
        }
        else
        {
            $(".plot").attr("src", "../LogisticRegression/"+directory+"/train.svg?" + new Date().getTime());
        }
    });
});
$(document).ready(function(){

    var t = 20;
    $("#t_size").on("change", function(){
            t = $("#t_size").val();
        });
    $("#select_data").on("change", function(){
            dataset = $("#select_data").val();
            if (dataset==="Social_Network_Ads.csv") directory = "SocialNetworkAds";
            else if (dataset === "moons") directory = "Moons";
            else if (dataset === "circles") directory = "Circles";

            $(".show_data").attr("href", "../LogisticRegression/" + directory + "/data.png");    
        });
    $(".run_button").click(function(){
        // alert(t);
        // alert(dataset);
        $.ajax({
                type: 'POST',
                url: "/cgi-bin/main.py",
                data: { algo:"logistic_regression", dataset:dataset, t_size:t}, //passing some input here
                dataType: "text",
                success: function(response){
                   output = response;
                   // alert(output);
                }
        	}).done(function(data){
            	console.log(data);
            	alert(data);
        	});
        $(".plot").attr("src", "../LogisticRegression/"+directory+"/test.svg?" + new Date().getTime());
        $.get("../LogisticRegression/"+directory+"/intercept.txt", function(data) {
         $("#intercept").text(data);});
        $.get("../LogisticRegression/"+directory+"/coef.txt", function(data) {
         $("#coef").text(data);});
        $.get("../LogisticRegression/"+directory+"/accu.txt", function(data) {
         $("#accu").text(data);});
        $.get("../LogisticRegression/"+directory+"/time.txt", function(data) {
         $("#etime").text(data);});
     });
});
