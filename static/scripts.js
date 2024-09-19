
function clearUrl(){
    document.getElementById('url-input').value='';
}

//ensure the DOM is fully loaded before attaching eventlisteners
document.addEventListener("DOMContentLoaded",function() {
    document.getElementById("clear-button").addEventListener("click",clearUrl);
});
