$(document).ready(function(){
    $("#traintest").on("change", function(){
        if($("#traintest").prop("checked"))
        {
            $(".plot").attr("src", "./SocialNetworkAds/test.svg");
        }
        else
        {
            $(".plot").attr("src", "./SocialNetworkAds/train.svg");
        }
    });
});
$(document).ready(function(){
  $("#Social_Network_Ads").click(function(){
    $(".plot").attr("src", "./SocialNetworkAds/test.svg");
  });
});
