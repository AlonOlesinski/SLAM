import numpy as np
import pickle
import matplotlib.pyplot as plt

from track_db_dir.track import Track
from shared_utils import compose_affine_transformations, read_cameras
from consensus_matching_dir.triangulation import triangulate_point


class TrackDB:
    def __init__(self):
        self.relative_R_t_list = []  # list of relative R_t
        self.global_R_t_list = []  # list of global R_t
        self.frameId_track_dict = {}  # value: dict of track_id: track
        # define two dictionaries according to which we start a new track / continue an existing track.
        self.last_added_tracks = {}  # key: x, y coordinates of the track in the last frame it appears in, value: track
        self.next_added_tracks = {}  # key: x, y coordinates of the track in the next frame it appears in, value: track
        self.inliers_outliers_count = [] # key: frame id, value: tuple of (inliers, outliers) count

    def get_relative_R_t(self, frame_id):
        """
        :param frame_id: frame id
        :return: relative R_t of the frame
        """
        return self.relative_R_t_list[frame_id]

    def get_global_R_t(self, frame_id):
        """
        :param frame_id: frame id
        :return: global R_t of the frame
        """
        return self.global_R_t_list[frame_id]

    def get_frame_tracks(self, frame_id):
        """
        :param frame_id: frame id
        :return: list of track ids that appear in the frame.
        """
        return self.frameId_track_dict[frame_id].values()

    def get_track_frames(self, track_id):
        """
        :param track_id: track id
        :return: list of frames that the track appears in.
        """
        frame = track_id[0]
        return self.frameId_track_dict[frame][track_id].get_frames()

    def add_match(self, l0_id: int, x_l0: int, y_l0: int, x_r0: int, y_r0: int,
                  l1_id: int, x_l1: int, y_l1: int, x_r1: int, y_r1: int):
        """
        :param l0_id: left0 track id
        :param l1_id: left1 track id
        :param x_l0: x coordinate of the track in left0
        :param y_l0: y coordinate of the track in left0
        :param x_r0: x coordinate of the track in right0
        :param y_r0: y coordinate of the track in right0
        :param x_l1: x coordinate of the track in left1
        :param y_l1: y coordinate of the track in left1
        :param x_r1: x coordinate of the track in right1
        :param y_r1: y coordinate of the track in right1
        """
        if l0_id not in self.frameId_track_dict.keys():
            self.frameId_track_dict[l0_id] = {}
        if l1_id not in self.frameId_track_dict.keys():
            self.frameId_track_dict[l1_id] = {}

        # check if the track already exists in the frame
        if (x_l0, y_l0) in self.last_added_tracks.keys():
            track = self.last_added_tracks[(x_l0, y_l0)]
        else:
            track = Track(l0_id, (x_l0, y_l0), (x_r0, y_r0))
            self.frameId_track_dict[l0_id][track.__id__()] = track

        track.add_location((x_l1, y_l1), (x_r1, y_r1))
        self.frameId_track_dict[l1_id][track.__id__()] = track
        self.next_added_tracks[(x_l1, y_l1)] = track

    def reset_last_added_tracks(self):
        """
        Restarts the two dictionaries that are used to determine if a track already exists in a frame.
        Should be used after each frame.
        :return:
        """
        self.last_added_tracks = self.next_added_tracks
        self.next_added_tracks = {}

    def get_location(self, frame_id, track_id):
        """
        :param frame_id: frame id
        :param track_id: track id
        :return: location of the track in the frame.
        """
        track = self.frameId_track_dict[frame_id][track_id]
        xl, y = track.locations[frame_id - track.first_frame_id][0]
        xr, _ = track.locations[frame_id - track.first_frame_id][1]
        return xl, xr, y

    def get_all_unique_tracks(self):
        """
        :return: list of all tracks in the track db.
        """
        tracks = []
        for track_dict in self.frameId_track_dict.values():
            for track in track_dict.values():
                tracks.append(track)

        return set(tracks)

    def get_all_unique_tracks_longer_than_given(self, min_length):
        """
        :return: list of all tracks in the track db.
        """
        tracks = []
        for track_dict in self.frameId_track_dict.values():
            for track in track_dict.values():
                if track.length() >= min_length:
                    tracks.append(track)

        return set(tracks)

    @staticmethod
    def serialize(track_db, filename):
        """
        :return: list of all tracks in the track db.
        """
        with open(filename, 'wb') as f:
            pickle.dump([track_db.frameId_track_dict,
                         track_db.inliers_outliers_count,
                         track_db.relative_R_t_list,
                         track_db.global_R_t_list], f)

    @staticmethod
    def deserialize(filename):
        """
        creates a track db from a serialized file
        :param filename: path to the serialized file
        :return: TrackDB object
        """
        track_db = TrackDB()
        with open(filename, 'rb') as f:
            loaded_list = pickle.load(f)
            track_db.frameId_track_dict = loaded_list[0]
            track_db.inliers_outliers_count = loaded_list[1]
            track_db.relative_R_t_list = loaded_list[2]
            track_db.global_R_t_list = loaded_list[3]

        return track_db

    def get_all_tracks_at_percentile_track_length(self, percentile):
        """
        :return: list of all tracks in the track db.
        """
        tracks = []
        for track_dict in self.frameId_track_dict.values():
            for track in track_dict.values():
                tracks.append(len(track))

        perc = np.percentile(tracks, percentile)
        tracks = []
        for track_dict in self.frameId_track_dict.values():
            for track in track_dict.values():
                if len(track) >= perc:
                    tracks.append(track)

        return set(tracks)

    def get_statistics(self):
        """
        Return the following statistics for:
        1. Number of tracks
        2. Number of frames
        3. Mean track length
        4. Mean number of frame links (number of tracks on an average frame)
        :return:
        """
        all_tracks = self.get_all_unique_tracks()
        num_tracks = len(all_tracks)
        mean_track_length = np.mean([len(track.locations) for track in all_tracks])
        num_frames = len(self.frameId_track_dict.keys())
        mean_num_frame_links = np.mean([len(self.frameId_track_dict[k]) for k in self.frameId_track_dict.keys()])
        return num_tracks, num_frames, mean_track_length, mean_num_frame_links

    def get_longest_track(self):
        """
        :return: the longest track in the track db
        """
        all_tracks = self.get_all_unique_tracks()
        max_track_length = np.max([len(track.locations) for track in all_tracks])
        for track in all_tracks:
            if len(track.locations) == max_track_length:
                return track

    def get_frame_connectivity(self, frame_id):
        """
        :param frame_id: frame id
        :return: number of tracks in the frame
        """
        track_dict = self.frameId_track_dict[frame_id]
        count = 0
        for track in track_dict.values():
            track_first_frame_id = track.first_frame_id
            if track_first_frame_id + len(track) - 1  > frame_id:
                count += 1
        return count

    def get_track_with_len_at_least_10(self):
        """
        :return: the first track with length at least 10
        """
        all_tracks = self.get_all_unique_tracks()
        for track in all_tracks:
            if len(track.locations) >= 10:
                return track

    def calc_global_R_t(self):
        """
        Fill the global R_t list by composing the matrices from the relative R_t list.
        :return:
        """
        for i in range(len(self.relative_R_t_list)):
            if i == 0:
                self.global_R_t_list.append(self.relative_R_t_list[i])
            else:
                self.global_R_t_list.append(compose_affine_transformations(self.relative_R_t_list[i],
                                                                            self.global_R_t_list[i - 1]))

    def get_number_of_frames(self):
        """
        :return: number of frames in the track db
        """
        return len(self.frameId_track_dict.keys())

    def get_median_track_length_from_frame(self, frame):
        """
        return the median track length, counting from the given frame for all frame links.
        """
        histogram = []
        for t in self.get_frame_tracks(frame):
            histogram.append(len(t) - (frame - t.first_frame_id))
        return np.ceil(np.percentile(histogram, q=70)).astype(int)

    def triangulate_from_last_frame(self, track):
        x_l, y_l = track.locations[-1][0]
        x_r, y_r = track.locations[-1][1]
        k, _, right0_extrinsic = read_cameras()
        left_camera_extrinsic_matrix = self.global_R_t_list[track.first_frame_id + len(track) - 1]
        right_camera_extrinsic_matrix = compose_affine_transformations(right0_extrinsic, left_camera_extrinsic_matrix)
        point_4d = triangulate_point(k @ left_camera_extrinsic_matrix,
                                  k @ right_camera_extrinsic_matrix, (x_l, y_l), (x_r, y_r))
        point_3d = point_4d[:3] / point_4d[3]
        return point_3d

    def plot_matches_per_frame(self, plot_path):
        """
        Plot the number of matches per frame
        """
        plt.clf()
        matches_per_frame = []
        for frame_id in self.frameId_track_dict.keys():
            matches_per_frame.append(len(self.frameId_track_dict[frame_id]))
        plt.plot(matches_per_frame)
        plt.plot([np.mean(matches_per_frame)] * len(matches_per_frame))
        plt.xlabel('Frame')
        plt.ylabel('Number of matches')
        plt.title('Number of matches per frame')
        plt.savefig(plot_path)
        plt.clf()

    def plot_inliers_percentage_per_frame(self, plot_path):
        """
        Plot the inliers percentage per frame
        """
        plt.clf()
        inliers_percentage_per_frame = []
        for frame_id in list(self.frameId_track_dict.keys())[:-1]:
            inliers_percentage_per_frame.append(self.inliers_outliers_count[frame_id][0] /
                                                (self.inliers_outliers_count[frame_id][0] +
                                                 self.inliers_outliers_count[frame_id][1]) * 100)
        plt.plot(inliers_percentage_per_frame)
        plt.plot([np.mean(inliers_percentage_per_frame)] * len(inliers_percentage_per_frame))
        plt.xlabel('Frame')
        plt.ylabel('Inliers percentage')
        plt.title('Inliers percentage per frame')
        plt.savefig(plot_path)
        plt.clf()