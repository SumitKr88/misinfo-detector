class DeepfakeDetectorLogic:
    def load_models(self):
        print("Loading Forensics & Ensemble Models...")
        from transformers import pipeline

        # Model 1: General Purpose AI Detector
        self.pipe1 = pipeline("image-classification", model="umm-maybe/AI-image-detector", device=0)

        # Model 2: The Specialist (Deepfake vs Real)
        self.pipe2 = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection", device=0)

        print("Production Ensemble Loaded.")

    def analyze_metadata(self, img_path):
        """
        Forensic Layer 1: EXIF Data
        Real photos usually have Camera Maker, Model, ISO, etc.
        AI images usually strip this or have none.
        """
        from PIL import Image, ExifTags
        try:
            img = Image.open(img_path)
            exif_data = img._getexif()

            if not exif_data:
                return False, "No Camera Metadata found (Suspicious for original files)"

            # Check for specific camera tags
            found_camera_tags = False
            details = []
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name in ['Make', 'Model', 'ISOSpeedRatings', 'DateTimeOriginal']:
                    found_camera_tags = True
                    details.append(f"{tag_name}: {str(value)[:20]}")

            if found_camera_tags:
                return True, f"Camera Signature Detected: {', '.join(details)}"
            else:
                return False, "Metadata present but lacks Camera Signature"

        except Exception:
            return False, "Metadata extraction failed"

    def analyze_frequency_domain(self, img_path):
        """
        Forensic Layer 2: Fast Fourier Transform (FFT)
        Real cameras produce a specific 'power law' falloff in frequency.
        AI Generators (GANs/Diffusion) often leave 'grid artifacts' or unnatural energy drops in the spectrum.
        """
        import cv2
        import numpy as np

        try:
            img = cv2.imread(img_path, 0) # Load grayscale

            # FFT Transform
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

            # Calculate Azimuthal Average (Energy distribution from center to edge)
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2

            # Analyze the High Frequency (Edge) Energy vs Low Frequency (Center)
            # Real photos have chaotic high-frequency noise.
            # AI (even with grain) often has structured or dampened high-freq energy.

            # Mask the center (Low Freq)
            mask_size = 30
            magnitude_spectrum[center_h-mask_size:center_h+mask_size, center_w-mask_size:center_w+mask_size] = 0

            avg_high_freq_energy = np.mean(magnitude_spectrum)

            # Thresholds (tuned for 1080p images)
            # AI images often have lower 'chaotic' energy than real sensors
            score_penalty = 0
            status = "Normal Spectrum"

            if avg_high_freq_energy < 85:
                score_penalty = 40
                status = "Abnormal Frequency Drop-off (Synthetic Smoothness)"
            elif avg_high_freq_energy > 160:
                # Too much uniform noise (artificial grain added)
                score_penalty = 20
                status = "Uniform Noise Pattern (Artificial Grain)"

            return score_penalty, f"Spectral Energy: {avg_high_freq_energy:.1f} ({status})"

        except Exception as e:
            return 0, f"FFT Failed: {str(e)}"

    def analyze_local_file(self, local_filename, file_type):
        details = []
        fake_probability = 0.0
        has_camera_data = False
        fft_penalty = 0

        if file_type.startswith("image"):
            try:
                # --- STEP 1: AI MODEL ENSEMBLE ---
                # Model 1 (General)
                res1 = self.pipe1(local_filename)
                m1_score = 0.0
                for item in res1:
                    if item['label'] in ['FAKE', 'AI', 'ARTIFICIAL']: m1_score = item['score'] * 100
                    elif item['label'] == 'REAL': m1_score = (1 - item['score']) * 100

                # Model 2 (Specialist)
                res2 = self.pipe2(local_filename)
                m2_score = 0.0
                for item in res2:
                    lbl = item['label'].lower()
                    if 'fake' in lbl or 'ai' in lbl: m2_score = item['score'] * 100
                    elif 'real' in lbl: m2_score = (1 - item['score']) * 100

                details.append(f"AI Detection Model A: {m1_score:.1f}% Fake Confidence")
                details.append(f"AI Detection Model B: {m2_score:.1f}% Fake Confidence")

                # --- STEP 2: DIGITAL FORENSICS ---

                # Metadata Check
                has_camera_data, meta_msg = self.analyze_metadata(local_filename)
                details.append(f"Metadata Analysis: {meta_msg}")

                # Frequency Domain Check (The Leonardo Killer)
                fft_penalty, fft_msg = self.analyze_frequency_domain(local_filename)
                details.append(f"Frequency Analysis: {fft_msg}")

                # --- STEP 3: SCORING LOGIC (STRICT MODE) ---

                # Start with the highest model score (Pessimistic approach)
                fake_probability = max(m1_score, m2_score)

                # HEURISTIC 1: The "Ghost" Rule
                # If there is NO metadata, the image loses "Benefit of the Doubt".
                if not has_camera_data:
                    fake_probability = max(fake_probability, 30) # Floor is now 30% Fake

                    # If Models are unsure (0-30%) BUT No Metadata + FFT Artifacts -> FLAG IT
                    if fake_probability < 40 and fft_penalty > 0:
                        fake_probability += fft_penalty
                        details.append("Pattern Match: Synthetic frequency patterns detected.")

                    # Heavy penalty for "No Metadata"
                    fake_probability += 20
                    details.append("Trust Penalty: Missing digital provenance (Metadata).")

                # HEURISTIC 2: The "Trust" Rule
                # If we have confirmed Camera Metadata (e.g. 'iPhone 13 Pro', 'ISO 80'), we trust it significantly
                elif has_camera_data and fake_probability < 80:
                    fake_probability -= 30
                    details.append("Trust Boost: Verified Camera Source.")

            except Exception as e:
                return {"status": "error", "message": f"Analysis failed: {str(e)}"}

        elif file_type.startswith("video"):
            fake_probability = 50.0
            details.append("Video analysis running in basic mode.")

        # --- FINAL VERDICT ---
        fake_probability = min(max(fake_probability, 0), 100)
        credibility_score = 100 - fake_probability

        verdict = "Likely Real"

        # Override Verdicts based on strict rules
        if credibility_score < 30:
            verdict = "AI Generated"
        elif credibility_score < 65: # Tightened threshold from 60 to 65
            verdict = "Suspicious / Unverified"
        elif not has_camera_data and credibility_score > 65:
            # CAP CREDIBILITY for non-metadata images
            credibility_score = 65
            verdict = "Unverified Source (No Metadata)"
            details.append("Result Capped: Cannot verify authenticity without metadata.")

        return {
            "status": "success",
            "score": credibility_score,
            "verdict": verdict,
            "details": details,
            "metadata": {"type": file_type}
        }
