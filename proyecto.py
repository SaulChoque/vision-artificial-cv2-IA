import cv2
import numpy as np

try:
    import pyopencl as cl  # type: ignore
except Exception:
    cl = None


KERNEL_SOBEL = """
__kernel void sobel(
    __global uchar* img,
    __global uchar* res,
    int ancho,
    int alto
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < 1 || x >= ancho-1 || y < 1 || y >= alto-1) return;

    int gx =
        -img[(y-1)*ancho + (x-1)] + img[(y-1)*ancho + (x+1)] +
        -2*img[y*ancho + (x-1)] + 2*img[y*ancho + (x+1)] +
        -img[(y+1)*ancho + (x-1)] + img[(y+1)*ancho + (x+1)];

    int gy =
        -img[(y-1)*ancho + (x-1)] - 2*img[(y-1)*ancho + x] - img[(y-1)*ancho + (x+1)] +
         img[(y+1)*ancho + (x-1)] + 2*img[(y+1)*ancho + x] + img[(y+1)*ancho + (x+1)];

    int mag = abs(gx) + abs(gy);
    if(mag > 255) mag = 255;
    res[y*ancho + x] = mag;
}
"""

KERNEL_CANNY = """
__kernel void canny(
    __global uchar* img,
    __global uchar* res,
    int ancho,
    int alto,
    uchar umbral
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < 1 || x >= ancho-1 || y < 1 || y >= alto-1) return;

    int gx =
        -img[(y-1)*ancho + (x-1)] + img[(y-1)*ancho + (x+1)] +
        -2*img[y*ancho + (x-1)] + 2*img[y*ancho + (x+1)] +
        -img[(y+1)*ancho + (x-1)] + img[(y+1)*ancho + (x+1)];

    int gy =
        -img[(y-1)*ancho + (x-1)] - 2*img[(y-1)*ancho + x] - img[(y-1)*ancho + (x+1)] +
         img[(y+1)*ancho + (x-1)] + 2*img[(y+1)*ancho + x] + img[(y+1)*ancho + (x+1)];

    int mag = abs(gx) + abs(gy);
    if(mag > 255) mag = 255;

    if(mag > umbral)
        res[y*ancho + x] = 255;
    else
        res[y*ancho + x] = 0;
}
"""


def _procesar_cpu(img_gray: np.ndarray, umbral: int = 200) -> tuple[np.ndarray, np.ndarray, str]:
    sobel_x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(cv2.addWeighted(np.abs(sobel_x).astype(np.uint8), 0.5, np.abs(sobel_y).astype(np.uint8), 0.5, 0))
    canny = cv2.Canny(img_gray, threshold1=max(10, umbral // 2), threshold2=umbral)
    return sobel, canny, "cpu"


def _procesar_opencl(img_gray: np.ndarray, umbral: int = 200) -> tuple[np.ndarray, np.ndarray, str]:
    if cl is None:
        raise RuntimeError("PyOpenCL no está disponible")

    alt, anc = img_gray.shape
    plataforma = cl.get_platforms()[0]
    dispositivo = plataforma.get_devices()[0]
    contexto = cl.Context([dispositivo])
    cola = cl.CommandQueue(contexto)

    prog_sobel = cl.Program(contexto, KERNEL_SOBEL).build()
    prog_canny = cl.Program(contexto, KERNEL_CANNY).build()

    buf_img = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=img_gray)
    buf_sobel = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, img_gray.nbytes)
    buf_canny = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, img_gray.nbytes)

    prog_sobel.sobel(cola, (anc, alt), None, buf_img, buf_sobel, np.int32(anc), np.int32(alt))
    prog_canny.canny(
        cola,
        (anc, alt),
        None,
        buf_img,
        buf_canny,
        np.int32(anc),
        np.int32(alt),
        np.uint8(umbral),
    )

    sobel_res = np.empty_like(img_gray)
    canny_res = np.empty_like(img_gray)
    cl.enqueue_copy(cola, sobel_res, buf_sobel)
    cl.enqueue_copy(cola, canny_res, buf_canny)
    cola.finish()
    return sobel_res, canny_res, "opencl"


def procesar_imagen(img_bgr: np.ndarray, umbral: int = 200, escala: float = 1.0) -> dict:
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("La imagen está vacía")

    if escala <= 0:
        raise ValueError("La escala debe ser mayor a 0")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if escala != 1.0:
        img_gray = cv2.resize(img_gray, None, fx=escala, fy=escala)

    try:
        sobel_res, canny_res, backend = _procesar_opencl(img_gray, umbral)
    except Exception:
        sobel_res, canny_res, backend = _procesar_cpu(img_gray, umbral)

    total_pixeles = img_gray.size
    bordes = int(np.count_nonzero(canny_res))
    densidad = float(bordes / total_pixeles) if total_pixeles else 0.0

    return {
        "gris": img_gray,
        "sobel": sobel_res,
        "canny": canny_res,
        "parametros": {
            "ancho": int(img_gray.shape[1]),
            "alto": int(img_gray.shape[0]),
            "umbral": int(umbral),
            "escala": float(escala),
            "pixeles_borde": bordes,
            "densidad_borde": round(densidad, 6),
            "backend": backend,
        },
    }


if __name__ == "__main__":
    imagen = cv2.imread("monedas.jpg")
    if imagen is None:
        print("No se pudo cargar monedas.jpg")
        raise SystemExit(1)

    resultado = procesar_imagen(imagen, umbral=200, escala=2.0)
    cv2.imshow("Original", cv2.resize(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2))
    cv2.imshow("Sobel", resultado["sobel"])
    cv2.imshow("Canny", resultado["canny"])
    cv2.waitKey(0)