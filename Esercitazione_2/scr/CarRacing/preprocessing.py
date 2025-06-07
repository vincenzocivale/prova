import numpy as np
import cv2

class ImageProcessor:
    def __init__(self, target_shape=(84, 84), grayscale=True, stack_frames=4):
        """
        Inizializza il processore di immagini.
        Args:
            target_shape (tuple): La dimensione a cui ridimensionare l'immagine.
            grayscale (bool): Se convertire l'immagine in scala di grigi.
            stack_frames (int): Numero di frame da impilare.
        """
        self.target_shape = target_shape
        self.grayscale = grayscale
        self.stack_frames = stack_frames
        self.frame_buffer = []

    def process(self, frame):
        """
        Applica le operazioni di pre-processing a un singolo frame.
        """
        # Ritaglia la parte inferiore (dove ci sono i punteggi)
        frame = frame[:84, :, :] # Per esempio, rimuove le ultime 12 righe di pixel

        # Converti in scala di grigi se richiesto
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Ridimensiona l'immagine
        frame = cv2.resize(frame, self.target_shape, interpolation=cv2.INTER_AREA)

        # Normalizza i pixel a [0, 1]
        frame = frame.astype(np.float32) / 255.0

        return frame

    def stack_observation(self, processed_frame, reset=False):
        """
        Impila i frame pre-processati per creare l'osservazione finale.
        Args:
            processed_frame (np.ndarray): Il frame già processato.
            reset (bool): Se resettare il buffer dei frame.
        """
        if reset:
            self.frame_buffer = []
            # Inizializza il buffer con il primo frame replicato stack_frames volte
            for _ in range(self.stack_frames):
                self.frame_buffer.append(processed_frame)
        else:
            self.frame_buffer.pop(0)
            self.frame_buffer.append(processed_frame)

        # Concatena i frame lungo il nuovo asse (canale)
        return np.stack(self.frame_buffer, axis=-1)

