render_title <- function(){
    var url = `${window.location.href}info`;
    fetch(url).then(res => res.json()).then(d => info = d)
}
