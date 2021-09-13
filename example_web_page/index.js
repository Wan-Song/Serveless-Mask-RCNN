$(document).ready(function() {
  //Url do google cloud functions
  serverlessURL = 'https://us-central1-even-gearbox-325502.cloudfunctions.net/new_maskrcnn';

  const toDataURL = url => fetch(url).then(
    response => response.blob()).then(
    blob => new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob)
    }));
  
  [
    '001'                                                               
  
  ].forEach(addCard);

  function addCard(fish, index, array) {
    $('body').append(
      `
      <div class="card" id="${fish}">
        <img src="fishs/${fish}.png" style="width: 300px; height: 300px;">
        <div class="container">
          <h4><b>ID: </b>${fish}</h4>
        </div>
        <div class="result">WAITING FOR RESULT</div>
        <div class="time"></div>
      </div>
      `
    );
    var img = $(`#${fish}`).children()[0];

    toDataURL(`fishs/${fish}.png`)
      .then(dataUrl => {
        var start = Date.now();
        axios({
          method: 'post',
          url: serverlessURL,
          data: Qs.stringify({
            image_base64: dataUrl.replace(/^data:image\/(png|jpg);base64,/, "")
          }),
          headers: {
            'Accept': 'application/x-www-form-urlencoded',
            'Content-Type': 'application/x-www-form-urlencoded'
          }
        }).then(function(response) {
          var data = JSON.parse(response.data.replace(/'/g,"\""));
          var tamanho = data['Tamanho']

          var miliseconds = Date.now() - start;
          $(`#${fish} div.result`).css("background-color", "green");
          $(`#${fish} div.result`).text(`${tamanho} cm`);
          $(`#${fish} div.time`).text(`${miliseconds} ms`);
        });
      });
  };
});