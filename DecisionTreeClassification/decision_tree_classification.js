var dataset = "";
var directory = "";
$(document).ready(function(){
    $("#traintest").on("change", function(){
        if($("#traintest").prop("checked"))
        {
            $(".plot").attr("src", "../DecisionTreeClassification/"+directory+"/test.svg?" + new Date().getTime());
        }
        else
        {
            $(".plot").attr("src", "../DecisionTreeClassification/"+directory+"/train.svg?" + new Date().getTime());
        }
    });
});
$(document).ready(function(){
    var t = 20;
    var t = 20;
    var d=1;
    var c="gini"
    var n=20;
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
    $("#select_data").on("change", function(){
        dataset = $("#select_data").val();
        if (dataset==="Social_Network_Ads.csv") directory = "SocialNetworkAds";
        else if (dataset === "moons") directory = "Moons";
        else if (dataset === "circles") directory = "Circles";
        $(".show_data").attr("href", "../DecisionTreeClassification/" + directory + "/data.png");  

    });
    var rfc = false;
    $(".rfc").click(function(){
        $(".rfc_toggle").toggle();
        // alert($(".rfc_toggle").toggle());
        if($('.rfc').data('clicked')) {
            $('.rfc').data('clicked', false)
        // alert('yes');
        rfc = false;
        }
        else{
            $('.rfc').data('clicked', true)
            // alert('no');
            rfc = true;
        }
    });

  $(".run_button").click(function(){
    // alert(d);
    // alert(t);
    // alert(c);

    if (rfc)
    {
        $.ajax({
            type: 'POST',
            url: "/cgi-bin/main.py",
            data: { algo:"random_forest_classification", t_size:t, dataset:dataset, n_estimators:n, max_depth:d, criterion:c }, //passing some input here
            dataType: "text",
            success: function(response){
               output = response;
               // alert(output);
            }
    	}).done(function(data){
        	console.log(data);
        	alert("rfc");
    	});
    }
    else {
        $.ajax({
                type: 'POST',
                url: "/cgi-bin/main.py",
                data: { algo:"decision_tree_classification", dataset:dataset, t_size:t, max_depth:d, criterion:c }, //passing some input here
                dataType: "text",
                success: function(response){
                   output = response;
                   // alert(output);
                }
        	}).done(function(data){
            	console.log(data);
            	alert(data);
        	});
    }

    $(".plot").attr("src", "../DecisionTreeClassification/"+directory+"/test.svg?" + new Date().getTime());
    $.get("../DecisionTreeClassification/"+directory+"/depth.txt", function(data) {
     $("#depth").text(data);});
    $.get("../DecisionTreeClassification/"+directory+"/n_leaves.txt", function(data) {
     $("#n_leaves").text(data);});
    $.get("../DecisionTreeClassification/"+directory+"/accu.txt", function(data) {
     $("#accu").text(data);});
    $.get("../DecisionTreeClassification/"+directory+"/time.txt", function(data) {
     $("#etime").text(data);});
    $(".show_cm").attr("href", "../svc/"+directory+"/cm.svg?" + new Date().getTime());
    $(".show_cr").attr("href", "../svc/"+directory+"/cr.svg?" + new Date().getTime());



 });
});
