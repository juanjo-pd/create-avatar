# Roadmap Técnico: Pipeline de Generación de Avatares (Busto)

**Objetivo:** Generar avatares semi-realistas de busto (hombros hacia arriba) con ARKit blendshapes + visemes, a partir de una foto del usuario, exportando en formato GLB.

**Escala objetivo:** Decenas de avatares (10-50), procesamiento en batch.

---

## Arquitectura General del Pipeline

```
Foto del usuario
       │
       ▼
┌──────────────────────┐
│  1. DETECCIÓN FACIAL  │  MediaPipe / dlib
│     + Preprocesado    │  (alineación, crop, normalización)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  2. RECONSTRUCCIÓN    │  DECA / EMOCA / MICA
│     3D DEL ROSTRO     │  (foto → parámetros FLAME)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  3. GENERACIÓN DE     │  Deformation Transfer
│     ARKIT BLENDSHAPES │  o mapping FLAME → ARKit
│     + VISEMES         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  4. TEXTURA FACIAL    │  UV unwrap + proyección
│                       │  desde la foto original
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  5. MONTAJE BUSTO     │  Cuello + hombros genéricos
│     + RIGGING         │  con esqueleto humanoide
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  6. EXPORT GLB        │  Blender headless
│     (batch)           │  bpy.ops.export_scene.gltf()
└──────────────────────┘
```

---

## Fase 1: Detección Facial y Preprocesado

**Qué hace:** Toma la foto del usuario, detecta el rostro, lo alinea y lo normaliza para que el reconstructor 3D funcione bien.

**Herramientas:**

| Herramienta | Repo | Notas |
|---|---|---|
| MediaPipe Face Mesh | Incluido en `mediapipe` (pip) | 468 landmarks, rápido, ideal para batch |
| dlib | `pip install dlib` | 68 landmarks, más establecido |
| MTCNN | `pip install mtcnn` | Bueno para detección + alineación |

**Recomendación:** MediaPipe. Es el más rápido, no necesita GPU para esta fase, y da 468 puntos faciales que se pueden usar después para validación.

**Script básico:**

```python
import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def preprocess_photo(image_path):
    img = cv2.imread(image_path)
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        raise ValueError(f"No se detectó rostro en {image_path}")

    # Crop y alineación basados en landmarks
    landmarks = results.multi_face_landmarks[0]
    # ... lógica de crop/align
    return aligned_face, landmarks
```

**Criterios de calidad del input:**
- Foto frontal o semi-frontal (hasta ~30° de rotación)
- Iluminación uniforme (evitar sombras duras)
- Resolución mínima: 512x512 px en la zona del rostro
- Sin oclusiones (gafas de sol, manos, mascarillas)

---

## Fase 2: Reconstrucción 3D del Rostro (FLAME)

**Qué hace:** Toma la foto preprocesada y genera los parámetros del modelo FLAME: forma (identidad), expresión, pose y textura.

### Opciones principales

#### Opción A: DECA (Recomendada para empezar)

- **Repo:** [github.com/yfeng95/DECA](https://github.com/yfeng95/DECA)
- **Paper:** SIGGRAPH 2021
- **Output:** Parámetros FLAME (shape, expression, pose) + malla 3D + textura UV
- **Ventaja:** Muy documentado, código estable, captura detalles faciales (arrugas)
- **Requisitos:**
    - Python 3.8+
    - PyTorch 1.9+ con CUDA
    - PyTorch3D 0.6+
    - ~4 GB VRAM para inferencia

```bash
# Instalación
git clone https://github.com/yfeng95/DECA
cd DECA
pip install -r requirements.txt
# Descargar modelos preentrenados (ver README)
```

```python
# Uso en batch
from decalib.deca import DECA
from decalib.utils.config import cfg

cfg.model.use_tex = True
deca = DECA(config=cfg, device='cuda')

for photo_path in photo_list:
    codedict = deca.encode(images_tensor)
    opdict = deca.decode(codedict)
    # opdict contiene: shape, vertices, landmarks, albedo, etc.
```

#### Opción B: EMOCA (Mejor expresiones emocionales)

- **Repo:** [github.com/radekd91/emoca](https://github.com/radekd91/emoca)
- **Paper:** CVPR 2022
- **Ventaja:** Mejor captura de expresiones faciales
- **Nota:** Más complejo de instalar, usa el framework INFERNO

#### Opción C: MICA (Mejor precisión métrica)

- **Ventaja:** La forma 3D es más fiel a las proporciones reales del rostro
- **Ideal si:** Necesitas que los avatares se parezcan mucho al usuario

#### Opción D: INFERNO/EMICA (Todo en uno)

- **Repo:** [github.com/radekd91/inferno](https://github.com/radekd91/inferno)
- **Combina:** DECA + EMOCA + MICA + SPECTRE
- **Ventaja:** Framework unificado, lo más completo
- **Desventaja:** Más pesado de configurar

### Sobre el modelo FLAME

Necesitas registrarte para descargar el modelo base:
- **Sitio:** [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/)
- **Licencia:** Gratuito para investigación, comercial requiere contactar a MPI
- **Importante:** Esto es un requisito para DECA, EMOCA y cualquier framework basado en FLAME

### Output de esta fase

Para cada foto procesada obtendrás:
- `shape_params` (100 dims): identidad facial del usuario
- `expression_params` (50 dims): expresión neutra base
- `pose_params` (6 dims): rotación de cuello y mandíbula
- `vertices` (5023 vértices): malla 3D del rostro
- `albedo_texture`: textura facial en espacio UV

---

## Fase 3: ARKit Blendshapes + Visemes

**Qué hace:** Convierte la malla FLAME neutral en 52 ARKit blendshapes + 15 visemes. Esta es la fase más crítica y la que más trabajo lleva.

### Estrategia recomendada: Deformation Transfer

El método más robusto para generar ARKit blendshapes sobre cualquier malla facial es **Deformation Transfer** (Sumner & Popović, 2004). La idea: tienes una malla de referencia con los 52 blendshapes ya hechos, y "transfieres" esas deformaciones a tu malla FLAME reconstruida.

**Repo clave:** [github.com/vasiliskatr/deformation_transfer_ARkit_blendshapes](https://github.com/vasiliskatr/deformation_transfer_ARkit_blendshapes)

Este repo implementa exactamente esto: toma cualquier malla 3D facial y genera los 52 ARKit blendshapes por deformation transfer.

### Los 52 ARKit Blendshapes

Referencia completa: [arkit-face-blendshapes.com](https://arkit-face-blendshapes.com/)

Grupos principales:
- **Ojos (14):** eyeBlinkLeft/Right, eyeLookDown/Up/In/Out Left/Right, eyeSquintLeft/Right, eyeWideLeft/Right
- **Cejas (4):** browDownLeft/Right, browInnerUp, browOuterUpLeft/Right
- **Nariz (2):** noseSneerLeft/Right
- **Boca (28):** mouthClose, mouthFunnel, mouthPucker, mouthLeft/Right, mouthSmileLeft/Right, mouthFrownLeft/Right, mouthDimpleLeft/Right, mouthStretchLeft/Right, mouthRollLower/Upper, mouthShrugLower/Upper, mouthPressLeft/Right, mouthLowerDownLeft/Right, mouthUpperUpLeft/Right
- **Mandíbula (4):** jawOpen, jawForward, jawLeft, jawRight
- **Mejillas (2):** cheekPuff, cheekSquintLeft/Right
- **Lengua (1):** tongueOut

### Los 15 Visemes para Lip-Sync

| Viseme | Fonema ejemplo | Descripción |
|--------|---------------|-------------|
| sil | (silencio) | Boca cerrada/relajada |
| PP | p, b, m | Labios juntos |
| FF | f, v | Labio inferior toca dientes superiores |
| TH | th | Lengua entre dientes |
| DD | t, d, n | Lengua toca paladar |
| kk | k, g | Parte trasera de la lengua |
| CH | ch, j, sh | Similar a sonrisa tensa |
| SS | s, z | Dientes casi juntos |
| nn | n, l | Boca ligeramente abierta |
| RR | r | Labios ligeramente redondeados |
| aa | a | Boca abierta |
| E | e | Boca semi-abierta, estirada |
| I | i | Boca casi cerrada, estirada |
| O | o | Labios redondeados |
| U | u | Labios muy redondeados |

### Generación de visemes

Los visemes se pueden derivar como **combinaciones de ARKit blendshapes**. Por ejemplo:
- `aa` ≈ jawOpen(0.7) + mouthFunnel(0.2)
- `O` ≈ jawOpen(0.4) + mouthFunnel(0.6) + mouthPucker(0.3)
- `PP` ≈ mouthClose(0.9) + mouthPressLeft(0.5) + mouthPressRight(0.5)
- `E` ≈ jawOpen(0.3) + mouthSmileLeft(0.4) + mouthSmileRight(0.4)

Puedes definir estos mappings en un JSON y generar los visemes programáticamente una vez que tengas los ARKit blendshapes.

### Alternativa: ARKitBlendshapeHelper (Blender Addon)

- **Repo:** [github.com/elijah-atkins/ARKitBlendshapeHelper](https://github.com/elijah-atkins/ARKitBlendshapeHelper)
- Addon de Blender que genera ARKit blendshapes a partir de un rig facial existente
- Útil como complemento o para validar resultados

---

## Fase 4: Textura Facial

**Qué hace:** Genera una textura realista del rostro a partir de la foto original.

### Enfoque con DECA

DECA ya genera una textura facial (`albedo`) en espacio UV como parte de su output. Esta textura es un buen punto de partida pero puede necesitar refinamiento:

```python
# DECA ya te da la textura
opdict = deca.decode(codedict)
albedo = opdict['albedo_images']  # Textura UV del rostro
```

### Mejoras de textura

Para calidad semi-realista, considera:

1. **Proyección directa:** Proyectar la foto original sobre la malla UV usando las correspondencias de vértices. Esto da la textura más fiel.

2. **Inpainting de zonas ocultas:** Las zonas no visibles en la foto (orejas, parte trasera del cuello) necesitan rellenarse. Puedes usar técnicas de inpainting o una textura genérica de piel.

3. **Normalización de iluminación:** Para que la textura sea neutra (sin sombras de la foto original), puedes usar redes como SfSNet o los propios coeficientes de iluminación de DECA para "des-iluminar" la textura.

---

## Fase 5: Montaje del Busto + Rigging

**Qué hace:** Une la cabeza reconstruida con un cuello y hombros genéricos, y aplica un esqueleto (rig) humanoide.

### Cuerpo base (cuello + hombros)

Como solo necesitas busto, puedes:

1. **Crear una malla base genérica** en Blender con cuello y hombros. Una sola malla sirve para todos los avatares — solo cambia la cabeza.

2. **Usar SMPL-X recortado:** El modelo SMPL-X ([smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/)) incluye cuerpo completo. Puedes recortarlo hasta los hombros y usarlo como base parametrizable.

### Unión cabeza-cuello

El punto crítico es la costura (seam) entre la malla FLAME de la cabeza y la malla del busto:

```python
# Pseudocódigo para unión en Blender (bpy)
# 1. Posicionar la cabeza FLAME sobre el cuello del busto
# 2. Alinear los vértices del borde inferior de FLAME con el borde superior del cuello
# 3. Hacer bridge edge loops o merge by distance
# 4. Suavizar la zona de unión con smooth vertex
```

La malla FLAME tiene un borde abierto en la zona del cuello que se puede usar como punto de costura.

### Esqueleto (Rig)

Para busto necesitas un esqueleto mínimo:

```
Hips (root)
└── Spine
    └── Spine1
        └── Spine2 (pecho)
            ├── LeftShoulder
            │   └── LeftUpperArm (opcional, solo si muestras hombros)
            ├── RightShoulder
            │   └── RightUpperArm (opcional)
            └── Neck
                └── Head
                    ├── LeftEye
                    ├── RightEye
                    └── Jaw
```

Este esqueleto es compatible con el estándar humanoid de Unity/Unreal.

### Rigging automático en Blender

```python
# Script de Blender para asignar pesos automáticos
import bpy

# Seleccionar malla y armature
mesh = bpy.data.objects['avatar_mesh']
armature = bpy.data.objects['avatar_armature']

mesh.select_set(True)
armature.select_set(True)
bpy.context.view_layer.objects.active = armature

# Parent con pesos automáticos
bpy.ops.object.parent_set(type='ARMATURE_AUTO')
```

---

## Fase 6: Exportación GLB en Batch

**Qué hace:** Exporta cada avatar como archivo .glb con malla, textura, esqueleto, blendshapes y visemes incluidos.

### Script de exportación headless

```bash
# Ejecutar Blender en modo background
blender -b --python export_avatar.py -- --input_dir ./avatars_raw --output_dir ./avatars_glb
```

```python
# export_avatar.py
import bpy
import sys
import os

argv = sys.argv
argv = argv[argv.index("--") + 1:]

input_dir = argv[argv.index("--input_dir") + 1]
output_dir = argv[argv.index("--output_dir") + 1]

for blend_file in os.listdir(input_dir):
    if not blend_file.endswith('.blend'):
        continue

    filepath = os.path.join(input_dir, blend_file)
    bpy.ops.wm.open_mainfile(filepath=filepath)

    output_path = os.path.join(
        output_dir,
        blend_file.replace('.blend', '.glb')
    )

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        export_draco_mesh_compression_enable=False,
        export_morph=True,           # Incluir blendshapes/shape keys
        export_morph_normal=True,    # Normales de blendshapes
        export_skins=True,           # Incluir skinning (esqueleto)
        export_animations=False,     # Sin animaciones (solo rig)
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
    )

    print(f"Exportado: {output_path}")
```

**Parámetros clave de exportación:**
- `export_morph=True`: Incluye los shape keys (blendshapes) en el GLB
- `export_skins=True`: Incluye el esqueleto y los pesos de skinning
- `export_format='GLB'`: Binario, todo en un solo archivo

---

## Requisitos de Hardware

| Componente | Mínimo | Recomendado |
|---|---|---|
| GPU | NVIDIA con 4 GB VRAM | NVIDIA con 8+ GB VRAM |
| RAM | 16 GB | 32 GB |
| Almacenamiento | 20 GB (modelos + datos) | SSD 50 GB |
| CPU | 4 cores | 8+ cores (batch processing) |
| CUDA | 11.1+ | 11.8+ o 12.x |

**Tiempo estimado por avatar** (con GPU 8GB):
- Preprocesado foto: ~0.5 seg
- Reconstrucción DECA: ~2-3 seg
- Generación blendshapes (deformation transfer): ~10-30 seg
- Montaje busto + rig (Blender headless): ~5-10 seg
- Export GLB: ~2-3 seg
- **Total: ~20-50 segundos por avatar**

Para 50 avatares: **~15-40 minutos** de procesamiento total.

---

## Stack Tecnológico Completo

### Dependencias Python

```txt
# requirements.txt
torch>=1.12.0
torchvision>=0.13.0
pytorch3d>=0.7.0
mediapipe>=0.10.0
opencv-python>=4.7.0
numpy>=1.23.0
scipy>=1.9.0
scikit-image>=0.19.0
trimesh>=3.15.0
Pillow>=9.2.0
chumpy>=0.70        # Para FLAME
```

### Software adicional

- **Blender 3.6+ LTS** (para procesamiento headless)
- **CUDA Toolkit 11.8+**
- **conda/mamba** (gestión de entorno recomendada)

### Modelos preentrenados a descargar

1. **FLAME model** → [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/) (registro requerido)
2. **DECA pretrained** → Instrucciones en el README del repo
3. **Face detector** (viene con MediaPipe, descarga automática)

---

## Repos Clave (Resumen)

| Componente | Repo | Licencia |
|---|---|---|
| FLAME (modelo base) | [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/) | Académica (comercial bajo acuerdo) |
| DECA (reconstrucción) | [github.com/yfeng95/DECA](https://github.com/yfeng95/DECA) | MIT-like |
| EMOCA (alternativa) | [github.com/radekd91/emoca](https://github.com/radekd91/emoca) | Académica |
| INFERNO (todo en uno) | [github.com/radekd91/inferno](https://github.com/radekd91/inferno) | Académica |
| Deformation Transfer → ARKit | [github.com/vasiliskatr/deformation_transfer_ARkit_blendshapes](https://github.com/vasiliskatr/deformation_transfer_ARkit_blendshapes) | MIT |
| ARKit Blendshape Helper | [github.com/elijah-atkins/ARKitBlendshapeHelper](https://github.com/elijah-atkins/ARKitBlendshapeHelper) | MIT |
| MediaPipe ↔ FLAME mapping | [github.com/PeizhiYan/mediapipe-blendshapes-to-flame](https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame) | MIT |
| FLAME Universe (índice) | [github.com/TimoBolkart/FLAME-Universe](https://github.com/TimoBolkart/FLAME-Universe) | — |
| Blender Python API (export) | [docs.blender.org/api](https://docs.blender.org/api/current/bpy.ops.export_scene.html) | GPL |

---

## Plan de Ejecución Sugerido

### Semana 1-2: Setup + Prueba de concepto

- [ ] Registrarse en FLAME y descargar el modelo
- [ ] Instalar DECA, correr con una foto de prueba
- [ ] Verificar que obtienes malla + textura correctas
- [ ] Probar deformation transfer con el repo de vasiliskatr

### Semana 3-4: Pipeline de blendshapes

- [ ] Implementar generación de los 52 ARKit blendshapes
- [ ] Definir mappings de visemes (JSON con pesos de blendshapes)
- [ ] Validar blendshapes importando en Unity/Unreal
- [ ] Test con ARKit face tracking en iOS

### Semana 5-6: Busto + rigging + textura

- [ ] Crear malla base de busto en Blender
- [ ] Automatizar unión cabeza-cuello con script bpy
- [ ] Implementar rigging automático
- [ ] Refinar pipeline de textura (proyección + inpainting)

### Semana 7-8: Batch processing + pulido

- [ ] Script maestro que orqueste todo el pipeline
- [ ] Procesamiento en batch de 10-50 fotos
- [ ] Control de calidad: validar cada GLB exportado
- [ ] Documentar el pipeline para mantenimiento

---

## Consideraciones de Licencia

**Atención:** FLAME y varios frameworks derivados tienen licencia académica. Para uso comercial:

- **FLAME:** Requiere acuerdo de licencia comercial con el Max Planck Institute. Contactar a los autores.
- **DECA:** Revisar su licencia específica (generalmente permite derivados no comerciales).
- **SMPL-X:** Misma situación que FLAME (MPI).

Si el proyecto es comercial, investiga las condiciones de licencia antes de invertir tiempo en el pipeline. La alternativa es usar modelos con licencias más permisivas o negociar licencias comerciales.

---

*Documento generado el 28 de marzo de 2026. Los repos y herramientas enlazados pueden haber cambiado — verificar versiones actuales antes de implementar.*