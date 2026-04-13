import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
import pytest

from RETHINKING_GENERALIZATION.modules.pixel_shuffle import process_dataset_pixel_shuffle
from RETHINKING_GENERALIZATION.modules.randomize_pixels import process_dataset as randomize_dataset

def create_identical_dummy_images(num_images=3, size=(32, 32, 3), output_dir=None):
    """
    Belirtilen dizinde hepsi birbiriyle BİREBİR AYNI olan dummy resimler oluşturur.
    Her piksel için rastgele fakat tüm resimlerde aynı olacak değerler seçilir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(99) 
    identical_data = np.random.randint(0, 255, size=size, dtype=np.uint8)
    
    saved_paths = []
    for i in range(num_images):
        img_path = output_dir / f"dummy_image_{i}.png"
        Image.fromarray(identical_data).save(img_path)
        saved_paths.append(img_path)
        
    return saved_paths

def test_pixel_shuffle_uniformity():
    """
    Tüm görsellerin AYNI karıştırma haritası ile karıştırılıp karıştırılmadığını test eder.
    Aynı içeriğe sahip 3 görsel oluşturup sonuçlarının da birbiriyle aynı olmasını bekler.
    (Makaledeki deneyin beklediği davranış)
    """
    base_dir = Path(__file__).resolve().parent.parent
    dataset_name = "pytest_uniformity"
    
    raw_dir = base_dir / "data" / "raw" / f"{dataset_name}_images"
    processed_dir = base_dir / "data" / "processed" / f"{dataset_name}_pixel_shuffle"
    
    try:
        create_identical_dummy_images(num_images=3, size=(32, 32, 3), output_dir=raw_dir)
        process_dataset_pixel_shuffle(dataset_name, file_extension="*.png", seed=42)
        
        processed_files = sorted(list(processed_dir.glob("*.png")))
        shuffled_imgs = [np.array(Image.open(f)) for f in processed_files]
        
        # Test: Çıktıların tümü BİRBİRİYLE BİREBİR AYNI olmalı
        for i in range(1, len(shuffled_imgs)):
            assert np.array_equal(shuffled_imgs[0], shuffled_imgs[i]), (
                "Hata: pixel_shuffle aynı görselleri farklı karıştırmış!"
            )
            
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(processed_dir, ignore_errors=True)

def test_randomize_pixels_independence():
    """
    randomize_pixels modülünün her görseli BAĞIMSIZ (farklı) bir harita 
    ile karıştırıp karıştırmadığını test eder. Aynı içeriğe sahip 3 görselin 
    sonuçlarının birbirinden FARKLI olmasını bekler.
    (Makaledeki beklenen davranışın TAM TERSİ, yani naif rastgelelik durumu)
    """
    base_dir = Path(__file__).resolve().parent.parent
    dataset_name = "pytest_independence"
    
    raw_dir = base_dir / "data" / "raw" / f"{dataset_name}_images"
    processed_dir = base_dir / "data" / "processed" / f"{dataset_name}_randomized"
    
    try:
        create_identical_dummy_images(num_images=3, size=(32, 32, 3), output_dir=raw_dir)
        
        # randomize_pixels modülünü %100 karıştırma ile çağırıyoruz.
        randomize_dataset(src_dir=raw_dir, dest_dir=processed_dir, percent=100, ext="*.png")
        
        processed_files = sorted(list(processed_dir.glob("*.png")))
        randomized_imgs = [np.array(Image.open(f)) for f in processed_files]
        
        # Test: Çıktıların üçü de birbirinden FARKLI olmalı. 
        # Eğer en az bir tanesi bile diğeriyle tamamen aynı çıkarsa test başarısız olur!
        assert not np.array_equal(randomized_imgs[0], randomized_imgs[1]), "Hata: 1. ve 2. görsel BİREBİR AYNI karıştırılmış!"
        assert not np.array_equal(randomized_imgs[0], randomized_imgs[2]), "Hata: 1. ve 3. görsel BİREBİR AYNI karıştırılmış!"
        assert not np.array_equal(randomized_imgs[1], randomized_imgs[2]), "Hata: 2. ve 3. görsel BİREBİR AYNI karıştırılmış!"
            
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(processed_dir, ignore_errors=True)
