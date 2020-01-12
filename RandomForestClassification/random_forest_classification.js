$(document).ready(function(){
    $("#traintest").on("change", function(){
        if($("#traintest").prop("checked"))
        {
            $(".plot").attr("src", "../RandomForestClassification/SocialNetworkAds/test.svg?" + new Date().getTime());
        }
        else
        {
            $(".plot").attr("src", "../RandomForestClassification/SocialNetworkAds/train.svg?" + new Date().getTime());
        }
    });
});
$(document).ready(function(){
    var n=10;
    var t = 20;
    var d=1;
    var c="gini"
    $("#t_size").on("change", function(){
            t = $("#t_size").val();
        });
    $("#n_estimators").on("change", function(){
            n = $("#n_estimators").val();
        });
    $("input[name='criterion']").on("change", function(){
            c = $("input[name='criterion']:checked").val();
        });
    $("#max_depth").on("change", function(){
            d = $("#max_depth").val();
        });

  $(".run_button").click(function(){
    // alert(d);
    // alert(t);
    // alert(c);
    $.ajax({
            type: 'POST',
            url: "/cgi-bin/main.py",
            data: { algo:"random_forest_classification", t_size:t, n_estimators:n, max_depth:d, criterion:c }, //passing some input here
            dataType: "text",
            success: function(response){
               output = response;
               // alert(output);
            }
    	}).done(function(data){
        	console.log(data);
        	alert(data);
    	});
    $(".plot").attr("src", "../RandomForestClassification/SocialNetworkAds/test.svg?" + new Date().getTime());
    // $.get("../DecisionTreeClassification/SocialNetworkAds/depth.txt", function(data) {
    //  $("#depth").text(data);});
    // $.get("../DecisionTreeClassification/SocialNetworkAds/n_leaves.txt", function(data) {
    //  $("#n_leaves").text(data);});
    $.get("../RandomForestClassification/SocialNetworkAds/accu.txt", function(data) {
     $("#accu").text(data);});
    $.get("../RandomForestClassification/SocialNetworkAds/time.txt", function(data) {
     $("#etime").text(data);});



 });
});
