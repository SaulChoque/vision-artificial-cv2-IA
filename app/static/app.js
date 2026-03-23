const form = document.getElementById("procesar-form");
const estado = document.getElementById("estado");
const resultados = document.getElementById("resultados");
const imgSobel = document.getElementById("img-sobel");
const imgCanny = document.getElementById("img-canny");
const parametros = document.getElementById("parametros");
const btnProcesar = document.getElementById("btn-procesar");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  estado.textContent = "Procesando imagen...";
  btnProcesar.disabled = true;

  const formData = new FormData(form);

  try {
    const response = await fetch("/api/procesar", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Error desconocido al procesar");
    }

    imgSobel.src = `data:image/png;base64,${data.imagenes.sobel}`;
    imgCanny.src = `data:image/png;base64,${data.imagenes.canny}`;
    parametros.textContent = JSON.stringify(data.parametros, null, 2);

    resultados.classList.remove("hidden");
    estado.textContent = "Imagen procesada correctamente.";
  } catch (error) {
    estado.textContent = `Error: ${error.message}`;
  } finally {
    btnProcesar.disabled = false;
  }
});
