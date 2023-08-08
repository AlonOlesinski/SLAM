
class Track:
    def __init__(self, first_frame_id, first_frame_location_left: tuple[int, int],
                 first_frame_location_right: tuple[int, int]):
        """
        :param track_id: tuple of (frame_id, x, y). frame_id is the first frame the track appears in. x, y are the pixel
        coordinates of the track in the first left image.
        """
        self.first_frame_id = first_frame_id

        # list of ((x_l, y_l),(x_r, y_r)) pixel coordinates of the track in the left and right
        # images.
        self.locations = [(first_frame_location_left, first_frame_location_right)]

    def __id__(self):
        """
        :return: the id of the track
        """
        return self.first_frame_id, self.locations[0][0]

    def __len__(self):
        """
        :return: the number of frames the track appears in
        """
        return len(self.locations)

    def add_location(self, location_left: tuple[int, int], location_right: tuple[int, int]):
        """
        add a location of the track in a new frame
        :param location_left:
        :param location_right:
        """
        self.locations.append((location_left, location_right))

    def get_frames(self):
        """
        :return: all frames that the track appears in.
        """
        return [self.first_frame_id + i for i in range(len(self.locations))]

    def get_frame_from_relative_index(self, relative_index):
        """
        :param relative_index: the index of the frame relative to the first frame the track appears in.
        :return: the frame id
        """
        return self.first_frame_id + relative_index

    def get_4_frame_ids_and_left_img_location(self):
        """
        :return 4 equally spaced frame ids and location on the left images of the track in those
        frames. The first frame id is the first frame the track appears in. The last frame id is the
        last frame the track appears in.
        """
        frame_ids = [self.first_frame_id]
        locations = [self.locations[0][0]]
        if len(self.locations) == 1:
            return frame_ids, locations
        frame_ids.append(self.first_frame_id + (len(self.locations) - 1) // 3)
        locations.append(self.locations[(len(self.locations) - 1) // 3][0])
        frame_ids.append(self.first_frame_id + (len(self.locations) - 1) // 3 * 2)
        locations.append(self.locations[(len(self.locations) - 1) // 3 * 2][0])
        frame_ids.append(self.first_frame_id + len(self.locations) - 1)
        locations.append(self.locations[-1][0])
        return frame_ids, locations

    def get_consecutive_frame_ids_and_locations(self, num_frames):
        """
        get num_frames consecutive frame ids and locations of the track in those frames. The first
        frame id is the first frame the track appears in.
        """
        assert len(self.locations) >= num_frames
        frame_ids = [self.first_frame_id]
        locations = [self.locations[0]]
        for i in range(1, num_frames):
            frame_ids.append(self.first_frame_id + i)
            locations.append(self.locations[i])

        return frame_ids, locations


