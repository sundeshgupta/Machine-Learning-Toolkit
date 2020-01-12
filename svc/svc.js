
var dataset = "";
var directory = "";
$(document).ready(function(){
    $("#traintest").on("change", function(){
        if($("#traintest").prop("checked"))
        {
            $(".plot").attr("src", "../svc/"+directory+"/test.svg?" + new Date().getTime());
        }
        else
        {
            $(".plot").attr("src", "../svc/"+directory+"/train.svg?" + new Date().getTime());
        }
    });
});
$(document).ready(function(){
    var t = 20;
    var c=1;
    var k = "linear";
    $("input[name='kernel']").on("change", function(){
            k = $("input[name='kernel']:checked").val();
        });
    $("#t_size").on("change", function(){
            t = $("#t_size").val();
        });
    $("#C").on("change", function(){
            c = $("#C").val();
        });
    $("#select_data").on("change", function(){
            dataset = $("#select_data").val();
            if (dataset==="Social_Network_Ads.csv") directory = "SocialNetworkAds";
            else if (dataset === "moons") directory = "Moons";
            else if (dataset === "circles") directory = "Circles";
            $(".show_data").attr("href", "../svc/" + directory + "/data.png");  

        });
  $(".run_button").click(function(){
    // alert(c);
    // alert(k);
    // alert(dataset);
    $.ajax({
            type: 'POST',
            url: "/cgi-bin/main.py",
            data: {kernel:k, algo:"svc", dataset:dataset, t_size:t, C:c }, //passing some input here
            dataType: "text",
            success: function(response){
               output = response;
               // alert(output);
            }
    	}).done(function(data){
        	console.log(data);
        	alert(data);
    	});
    $(".plot").attr("src", "../svc/"+directory+"/test.svg?" + new Date().getTime());
    $.get("../svc/"+directory+"/intercept.txt", function(data) {
     $("#intercept").text(data);});
    $.get("../svc/"+directory+"/coef.txt", function(data) {
     $("#coef").text(data);});
    $.get("../svc/"+directory+"/accu.txt", function(data) {
     $("#accu").text(data);});
    $.get("../svc/"+directory+"/time.txt", function(data) {
     $("#etime").text(data);});
     $(".show_3d").attr("href", "../svc/"+directory+"/3d.svg?" + new Date().getTime());
     $(".show_cm").attr("href", "../svc/"+directory+"/cm.svg?" + new Date().getTime());
     $(".show_cr").attr("href", "../svc/"+directory+"/cr.svg?" + new Date().getTime());



 });
});
