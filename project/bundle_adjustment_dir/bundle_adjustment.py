import gtsam
import tqdm
import matplotlib.pyplot as plt

from bundle_adjustment_dir.local_bundle import LocalBundle
from shared_utils import gtsam_calib_mat


class BundleAdjustment:

    def __init__(self, track_db, location_prior_sigma):
        self.track_db = track_db
        self.location_prior_sigma = location_prior_sigma
        self.bundles = []

    def set_bundles(self):
        """
        set the local bundles using the following heuristic. Start from the first frame and add the
        median (or other percentile) track length to the bundle. Minimum bundle length is 5 frames,
        maximum bundle length is 20 frames.
        """
        start_kf = 0
        max_frame = self.track_db.get_number_of_frames() - 1
        while start_kf < max_frame:
            end_kf = start_kf + self.track_db.get_median_track_length_from_frame(start_kf)
            if end_kf < start_kf + 4:
                end_kf = start_kf + 4
            if end_kf > start_kf + 20:
                end_kf = start_kf + 20
            if end_kf > max_frame:
                end_kf = max_frame
            bundle = LocalBundle(self.track_db, start_kf, end_kf, self.location_prior_sigma)
            bundle.create_factor_graph()
            self.bundles.append(bundle)
            start_kf = end_kf

    def print_key_frames(self):
        """
        print the key frames of the bundles
        """
        prev_bundle = None
        for bundle in self.bundles:
            print(
                f"start kf: {bundle.start_kf_idx}, end kf:  {bundle.end_kf_idx}, number of landmarks: {bundle.get_track_num()}")
            if prev_bundle is not None:
                assert prev_bundle.end_kf_idx == bundle.start_kf_idx
            prev_bundle = bundle

    def get_key_frames_indices(self):
        """
        :return: list of key frames, as indices in the full track.
        """
        key_frames = [0]
        for bundle in self.bundles:
            key_frames.append(bundle.end_kf_idx)
        return key_frames


    def get_bundle_lengths(self):
        """
        :return: list of bundle lengths
        """
        return [bundle.end_kf_idx - bundle.start_kf_idx + 1 for bundle in self.bundles]

    def run_local_adjustments(self, print_final_bundle_properties=False, return_error_per_bundle=False):
        """
        run levenberg marquardt optimization on all the bundles
        :param print_final_bundle_properties: print the location of the last bundle's first frame
         after optimization
        """
        error_before = 0
        error_after = 0
        mean_error_per_bundle_before_opt = []
        median_error_per_bundle_before_opt = []
        mean_error_per_bundle_after_opt = []
        median_error_per_bundle_after_opt = []
        for bundle in tqdm.tqdm(self.bundles, total=len(self.bundles)):
            error_before += bundle.get_error()
            # calculate error per factor before optimization:
            if return_error_per_bundle:
                error_per_factor = []
                for i in range(bundle.graph.size()):
                    error_per_factor.append(bundle.graph.at(i).error(bundle.values))
                mean_error_per_bundle_before_opt.append(sum(error_per_factor) / len(error_per_factor))
                median_error_per_bundle_before_opt.append(sorted(error_per_factor)[len(error_per_factor) // 2])

            bundle.optimize()

            # calculate error per factor after optimization:
            if return_error_per_bundle:
                error_per_factor = []
                for i in range(bundle.graph.size()):
                    error_per_factor.append(bundle.graph.at(i).error(bundle.values))
                mean_error_per_bundle_after_opt.append(sum(error_per_factor) / len(error_per_factor))
                median_error_per_bundle_after_opt.append(sorted(error_per_factor)[len(error_per_factor) // 2])

            if bundle.end_kf_idx == self.track_db.get_number_of_frames() - 1 \
                    and print_final_bundle_properties:
                print(f'location of the first frame after optimization:'
                      f' {bundle.values.atPose3(gtsam.symbol("c", bundle.start_kf_idx)).translation()}')
                first_factor = bundle.graph.at(0)
                print(
                    f'the anchoring factor error of the first frame after optimization: {first_factor.error(bundle.values)}')
            error_after += bundle.get_error()
        print("error before: %f, error after: %f" % (error_before, error_after))
        print("all bundles are optimized")

        if return_error_per_bundle:
            return mean_error_per_bundle_before_opt, median_error_per_bundle_before_opt, \
                   mean_error_per_bundle_after_opt, median_error_per_bundle_after_opt

    def set_global_coordinates(self):
        """
        set the global coordinates of the landmarks and cameras. This should be done after all the
         local adjustments
        """
        self.bundles[0].transformed_values = self.bundles[0].values
        for i in range(1, len(self.bundles)):
            new_values = gtsam.Values()
            prev_bundle = self.bundles[i - 1]
            cur_bundle = self.bundles[i]
            prev_last_pose = prev_bundle.transformed_values.atPose3(
                gtsam.symbol("c", prev_bundle.end_kf_idx))
            # transform cameras
            for j in range(self.bundles[i].start_kf_idx, self.bundles[i].end_kf_idx + 1):
                sym = gtsam.symbol("c", j)
                new_values.insert(sym, prev_last_pose * cur_bundle.values.atPose3(sym))
            # transform landmarks
            for j in cur_bundle.landmarks_idx:
                sym = gtsam.symbol("q", j)
                point = cur_bundle.values.atPoint3(sym)
                point = prev_last_pose.transformFrom(point)
                new_values.insert(sym, point)
            cur_bundle.transformed_values = new_values

    def get_all_camera_poses_global(self):
        """
        :return: a list of all camera poses in the global coordinates
        """
        poses = []
        for bundle in self.bundles:
            for i in range(bundle.start_kf_idx, bundle.end_kf_idx):
                poses.append(bundle.transformed_values.atPose3(gtsam.symbol("c", i)))
        poses.append(self.bundles[-1].transformed_values.atPose3(
            gtsam.symbol("c", self.bundles[-1].end_kf_idx)))
        return poses

    def get_all_landmarks_global(self):
        """
        :return: a list of all landmarks in the global coordinates
        """
        landmarks = []
        for bundle in self.bundles:
            for i in bundle.landmarks_idx:
                landmarks.append(bundle.transformed_values.atPoint3(gtsam.symbol("q", i)))
        return landmarks

    def get_key_frames_list(self):
        """
        :return: a list of all key frames indices
        """
        kf_list = []
        for bundle in self.bundles:
            kf_list.append(bundle.start_kf_idx)
        kf_list.append(self.bundles[-1].end_kf_idx)
        return kf_list

    def bundle_length_hist(self, plot_path):
        """
        plot the histogram of the bundle lengths
        """
        bundle_lengths = [bundle.end_kf_idx - bundle.start_kf_idx + 1 for bundle in self.bundles]
        plt.hist(bundle_lengths, bins=16)
        plt.savefig(plot_path)
        plt.clf()

    def get_reprojection_errors(self, min_distance=5):
        """
        project all the landmarks from the last frame they were seen in the bundle to the previous
        frames. calculate the reprojection error of each landmark in each frame and return a
        histogram of the errors (for both left and right cameras).
        :param bundle_idx: index of the bundle
        :return: reprojection error of the bundle
        """
        distance_error_dict = {} # key: distance from the projection frame, value: list of errors
        for bundle in self.bundles:
            for land_mark_sym, _ in bundle.get_all_landmarks_point3_gen(with_sym=True):
                camera_syms, frames = bundle.get_all_syms_of_cameras_with_factor_to_landmark(land_mark_sym, min_distance)
                if frames is None:
                    continue
                for i in range(len(frames)):
                    error = bundle.reprojection_error_between_landmark_and_camera(camera_syms[i],land_mark_sym)
                    distance = frames[i] - frames[0]
                    if distance not in distance_error_dict:
                        distance_error_dict[distance] = []
                    distance_error_dict[distance].append(error)
        return distance_error_dict

    def get_factor_errors(self, min_distance=5):
        distance_error_dict = {}  # key: distance from the projection frame, value: list of errors
        for bundle in self.bundles:
            for land_mark_sym, _ in bundle.get_all_landmarks_point3_gen(with_sym=True):
                camera_syms, frames = bundle.get_all_syms_of_cameras_with_factor_to_landmark(
                    land_mark_sym, min_distance)
                if frames is None:
                    continue
                for i in range(len(frames)):
                    error = bundle.factor_error_between_landmark_and_camera(camera_syms[i], land_mark_sym)
                    distance = frames[i] - frames[0]
                    if distance not in distance_error_dict:
                        distance_error_dict[distance] = []
                    distance_error_dict[distance].append(error)
        return distance_error_dict




